#include "color_conversion.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

bool is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaSuccess && count > 0) {
        return true;
    }
    return false;
}

// GPU float32 raw image cache singleton
static float* g_cached_device_raw_float_buf = nullptr;
static int g_cached_device_w = 0;
static int g_cached_device_h = 0;

void clear_cuda_device_cache() {
    if (g_cached_device_raw_float_buf) {
        cudaFree(g_cached_device_raw_float_buf);
        g_cached_device_raw_float_buf = nullptr;
    }
    g_cached_device_w = 0;
    g_cached_device_h = 0;
}

// Inline operators for float3 math inside CUDA device code
__device__ inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__global__ void convert_uint16_to_float_kernel(const uint16_t* input, float* output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx] / 65535.0f;
    }
}

/*
 * CUDA Color Conversion Kernel Design:
 * - Manually processes raw/crosstalk corrected sensor values.
 * - Extracts and applies the film profile's multi-stage tone curves (TRC) and 3D cLUT.
 * - Assumes the Profile Connection Space (PCS) is D50 XYZ (with 65535/32768 scaling).
 * - Manually applies chromatic adaptation (Bradford D50 -> D65) and XYZ-to-sRGB projection matrix.
 * - Applies a predefined/hardcoded sRGB gamma curve (piecewise or linear).
 * - Custom output profiles are NOT used or supported here; they must fallback to CPU LCMS.
 */
__global__ void color_conversion_kernel(
    const float* input_pixels,
    uint16_t* output_pixels,
    int w, int h,
    // Crosstalk matrix
    float cc0, float cc1, float cc2,
    float cc3, float cc4, float cc5,
    float cc6, float cc7, float cc8,
    // Film profile stages presence flags
    int has_profile,
    // Input TRC curves
    const float* in_trc_curve_0, int in_trc_size_0,
    const float* in_trc_curve_1, int in_trc_size_1,
    const float* in_trc_curve_2, int in_trc_size_2,
    // Output TRC curves
    const float* out_trc_curve_0, int out_trc_size_0,
    const float* out_trc_curve_1, int out_trc_size_1,
    const float* out_trc_curve_2, int out_trc_size_2,
    // Matrix stage
    float m0, float m1, float m2,
    float m3, float m4, float m5,
    float m6, float m7, float m8,
    float offset_0, float offset_1, float offset_2,
    // CLUT stage
    const float* clut_grid,
    int clut_dim_r, int clut_dim_g, int clut_dim_b,
    // Bradford adaptation matrix
    float ba0, float ba1, float ba2,
    float ba3, float ba4, float ba5,
    float ba6, float ba7, float ba8,
    // XYZ to sRGB matrix
    float srgb0, float srgb1, float srgb2,
    float srgb3, float srgb4, float srgb5,
    float srgb6, float srgb7, float srgb8,
    // Colorspace flag
    int colorspace_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;
 
    // 1. Load float32 raw pixels [0.0, 1.0]
    float r = input_pixels[idx * 3];
    float g = input_pixels[idx * 3 + 1];
    float b = input_pixels[idx * 3 + 2];

    // 2. Crosstalk matrix correction
    float cr = r * cc0 + g * cc1 + b * cc2;
    float cg = r * cc3 + g * cc4 + b * cc5;
    float cb = r * cc6 + g * cc7 + b * cc8;
    
    // Clip to [0.0, 1.0]
    cr = fmaxf(0.0f, fminf(1.0f, cr));
    cg = fmaxf(0.0f, fminf(1.0f, cg));
    cb = fmaxf(0.0f, fminf(1.0f, cb));

    if (has_profile) {
        // Lambda for 1D interpolation
        auto interpolate_trc = [](float val, const float* curve, int size) -> float {
            if (size <= 0 || !curve) return val;
            float scaled = val * (size - 1);
            int k = (int)floorf(scaled);
            k = max(0, min(k, size - 2));
            float delta = scaled - k;
            return curve[k] * (1.0f - delta) + curve[k + 1] * delta;
        };

        // 3. AtoB0 - 1D Input curves (TRC) interpolation
        cr = interpolate_trc(cr, in_trc_curve_0, in_trc_size_0);
        cg = interpolate_trc(cg, in_trc_curve_1, in_trc_size_1);
        cb = interpolate_trc(cb, in_trc_curve_2, in_trc_size_2);

        // 4. AtoB0 - 3x3 Matrix + Offset
        float mr = cr * m0 + cg * m1 + cb * m2 + offset_0;
        float mg = cr * m3 + cg * m4 + cb * m5 + offset_1;
        float mb = cr * m6 + cg * m7 + cb * m8 + offset_2;

        // Clip intermediate coordinates to [0.0, 1.0] for grid lookup
        mr = fmaxf(0.0f, fminf(1.0f, mr));
        mg = fmaxf(0.0f, fminf(1.0f, mg));
        mb = fmaxf(0.0f, fminf(1.0f, mb));

        // 5. AtoB0 - 3D CLUT tetrahedral interpolation
        float scaled_r = mr * (clut_dim_r - 1);
        float scaled_g = mg * (clut_dim_g - 1);
        float scaled_b = mb * (clut_dim_b - 1);

        int rf = (int)floorf(scaled_r);
        int gf = (int)floorf(scaled_g);
        int bf = (int)floorf(scaled_b);
        int rc = min(rf + 1, clut_dim_r - 1);
        int gc = min(gf + 1, clut_dim_g - 1);
        int bc = min(bf + 1, clut_dim_b - 1);
        rf = max(0, min(rf, clut_dim_r - 1));
        gf = max(0, min(gf, clut_dim_g - 1));
        bf = max(0, min(bf, clut_dim_b - 1));

        float dr = scaled_r - rf;
        float dg = scaled_g - gf;
        float db = scaled_b - bf;

        auto get_val = [=](int r_idx, int g_idx, int b_idx) -> float3 {
            int i = (r_idx * clut_dim_g * clut_dim_b + g_idx * clut_dim_b + b_idx) * 3;
            return make_float3(clut_grid[i], clut_grid[i+1], clut_grid[i+2]);
        };

        float3 v000 = get_val(rf, gf, bf);
        float3 v100 = get_val(rc, gf, bf);
        float3 v010 = get_val(rf, gc, bf);
        float3 v110 = get_val(rc, gc, bf);
        float3 v001 = get_val(rf, gf, bc);
        float3 v101 = get_val(rc, gf, bc);
        float3 v011 = get_val(rf, gc, bc);
        float3 v111 = get_val(rc, gc, bc);

        float3 clut_res;
        if (dr >= dg && dg >= db) {
            clut_res = v000 * (1.0f - dr) + v100 * (dr - dg) + v110 * (dg - db) + v111 * db;
        } else if (dr >= db && db > dg) {
            clut_res = v000 * (1.0f - dr) + v100 * (dr - db) + v101 * (db - dg) + v111 * dg;
        } else if (db > dr && dr >= dg) {
            clut_res = v000 * (1.0f - db) + v001 * (db - dr) + v101 * (dr - dg) + v111 * dg;
        } else if (dg > dr && dr >= db) {
            clut_res = v000 * (1.0f - dg) + v010 * (dg - dr) + v110 * (dr - db) + v111 * db;
        } else if (dg >= db && db > dr) {
            clut_res = v000 * (1.0f - dg) + v010 * (dg - db) + v011 * (db - dr) + v111 * dr;
        } else {
            clut_res = v000 * (1.0f - db) + v001 * (db - dg) + v011 * (dg - dr) + v111 * dr;
        }

        cr = clut_res.x;
        cg = clut_res.y;
        cb = clut_res.z;

        // 6. AtoB0 - 1D Output curves (TRC) interpolation
        cr = interpolate_trc(cr, out_trc_curve_0, out_trc_size_0);
        cg = interpolate_trc(cg, out_trc_curve_1, out_trc_size_1);
        cb = interpolate_trc(cb, out_trc_curve_2, out_trc_size_2);

        // Apply PCS scale correction: (65535.0f / 32768.0f)
        float scale_pcs = 65535.0f / 32768.0f;
        cr *= scale_pcs;
        cg *= scale_pcs;
        cb *= scale_pcs;
    }

    // 7. Transform D50 PCS XYZ to Output Color Space
    // Bradford adaptation (D50 -> D65)
    float xr = cr * ba0 + cg * ba1 + cb * ba2;
    float xg = cr * ba3 + cg * ba4 + cb * ba5;
    float xb = cr * ba6 + cg * ba7 + cb * ba8;

    // XYZ to Linear sRGB Matrix Projection
    float lr = xr * srgb0 + xg * srgb1 + xb * srgb2;
    float lg = xr * srgb3 + xg * srgb4 + xb * srgb5;
    float lb = xr * srgb6 + xg * srgb7 + xb * srgb8;

    // Clip to [0.0, 1.0]
    lr = fmaxf(0.0f, fminf(1.0f, lr));
    lg = fmaxf(0.0f, fminf(1.0f, lg));
    lb = fmaxf(0.0f, fminf(1.0f, lb));

    // 8. Non-linear sRGB mapping
    float out_r, out_g, out_b;
    if (colorspace_type == 0) { // sRGB piecewise EOTF
        out_r = (lr <= 0.0031308f) ? (lr * 12.92f) : (powf(lr, 1.0f / 2.4f) * 1.055f - 0.055f);
        out_g = (lg <= 0.0031308f) ? (lg * 12.92f) : (powf(lg, 1.0f / 2.4f) * 1.055f - 0.055f);
        out_b = (lb <= 0.0031308f) ? (lb * 12.92f) : (powf(lb, 1.0f / 2.4f) * 1.055f - 0.055f);
    } else { // Linear sRGB-g10
        out_r = lr;
        out_g = lg;
        out_b = lb;
    }

    // 9. Round to uint16
    output_pixels[idx * 3]     = (uint16_t)roundf(out_r * 65535.0f);
    output_pixels[idx * 3 + 1] = (uint16_t)roundf(out_g * 65535.0f);
    output_pixels[idx * 3 + 2] = (uint16_t)roundf(out_b * 65535.0f);
}

bool run_cuda_color_pipeline(
    const uint16_t* host_input_pixels,
    uint16_t* host_output_pixels,
    int w, int h,
    const float* crosstalk_matrix,
    float exposure_comp,
    int has_profile,
    const float* in_trc_curve_0, int in_trc_size_0,
    const float* in_trc_curve_1, int in_trc_size_1,
    const float* in_trc_curve_2, int in_trc_size_2,
    const float* out_trc_curve_0, int out_trc_size_0,
    const float* out_trc_curve_1, int out_trc_size_1,
    const float* out_trc_curve_2, int out_trc_size_2,
    const float* matrix_3x3,
    const float* offset_3,
    const float* clut_grid,
    int clut_dim_r, int clut_dim_g, int clut_dim_b,
    const float* bradford_matrix,
    const float* xyz_to_srgb_matrix,
    int colorspace_type
) {
    float* d_input_float = nullptr;
    uint16_t* d_output = nullptr;
    float* d_in_trc0 = nullptr;
    float* d_in_trc1 = nullptr;
    float* d_in_trc2 = nullptr;
    float* d_out_trc0 = nullptr;
    float* d_out_trc1 = nullptr;
    float* d_out_trc2 = nullptr;
    float* d_clut = nullptr;

    size_t img_size = w * h * 3 * sizeof(uint16_t);

    if (cudaMalloc(&d_output, img_size) != cudaSuccess) {
        return false;
    }

    // Check device cache singleton
    if (g_cached_device_raw_float_buf && g_cached_device_w == w && g_cached_device_h == h) {
        d_input_float = g_cached_device_raw_float_buf;
    } else {
        clear_cuda_device_cache();

        size_t float_img_size = w * h * 3 * sizeof(float);
        if (cudaMalloc(&g_cached_device_raw_float_buf, float_img_size) != cudaSuccess) {
            std::cerr << "ERROR: Failed to allocate device memory for float32 raw image cache." << std::endl;
            cudaFree(d_output);
            return false;
        }

        uint16_t* d_temp_in = nullptr;
        if (cudaMalloc(&d_temp_in, img_size) != cudaSuccess) {
            std::cerr << "ERROR: Failed to allocate temporary device memory for uint16 raw image." << std::endl;
            cudaFree(d_output);
            cudaFree(g_cached_device_raw_float_buf);
            g_cached_device_raw_float_buf = nullptr;
            return false;
        }

        if (cudaMemcpy(d_temp_in, host_input_pixels, img_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "ERROR: Failed to copy host raw image to device." << std::endl;
            cudaFree(d_temp_in);
            cudaFree(d_output);
            cudaFree(g_cached_device_raw_float_buf);
            g_cached_device_raw_float_buf = nullptr;
            return false;
        }

        int threads_per_block = 256;
        int blocks = (w * h * 3 + threads_per_block - 1) / threads_per_block;
        convert_uint16_to_float_kernel<<<blocks, threads_per_block>>>(d_temp_in, g_cached_device_raw_float_buf, w * h * 3);

        cudaFree(d_temp_in);

        g_cached_device_w = w;
        g_cached_device_h = h;
        d_input_float = g_cached_device_raw_float_buf;
    }

    if (has_profile) {
        if (in_trc_size_0 > 0 && in_trc_curve_0) {
            cudaMalloc(&d_in_trc0, in_trc_size_0 * sizeof(float));
            cudaMemcpy(d_in_trc0, in_trc_curve_0, in_trc_size_0 * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (in_trc_size_1 > 0 && in_trc_curve_1) {
            cudaMalloc(&d_in_trc1, in_trc_size_1 * sizeof(float));
            cudaMemcpy(d_in_trc1, in_trc_curve_1, in_trc_size_1 * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (in_trc_size_2 > 0 && in_trc_curve_2) {
            cudaMalloc(&d_in_trc2, in_trc_size_2 * sizeof(float));
            cudaMemcpy(d_in_trc2, in_trc_curve_2, in_trc_size_2 * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (out_trc_size_0 > 0 && out_trc_curve_0) {
            cudaMalloc(&d_out_trc0, out_trc_size_0 * sizeof(float));
            cudaMemcpy(d_out_trc0, out_trc_curve_0, out_trc_size_0 * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (out_trc_size_1 > 0 && out_trc_curve_1) {
            cudaMalloc(&d_out_trc1, out_trc_size_1 * sizeof(float));
            cudaMemcpy(d_out_trc1, out_trc_curve_1, out_trc_size_1 * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (out_trc_size_2 > 0 && out_trc_curve_2) {
            cudaMalloc(&d_out_trc2, out_trc_size_2 * sizeof(float));
            cudaMemcpy(d_out_trc2, out_trc_curve_2, out_trc_size_2 * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (clut_grid && clut_dim_r > 0 && clut_dim_g > 0 && clut_dim_b > 0) {
            size_t clut_size = clut_dim_r * clut_dim_g * clut_dim_b * 3 * sizeof(float);
            cudaMalloc(&d_clut, clut_size);
            cudaMemcpy(d_clut, clut_grid, clut_size, cudaMemcpyHostToDevice);
        }
    }

    int threads_per_block = 256;
    int blocks = (w * h + threads_per_block - 1) / threads_per_block;

    color_conversion_kernel<<<blocks, threads_per_block>>>(
        d_input_float, d_output, w, h,
        crosstalk_matrix[0], crosstalk_matrix[1], crosstalk_matrix[2],
        crosstalk_matrix[3], crosstalk_matrix[4], crosstalk_matrix[5],
        crosstalk_matrix[6], crosstalk_matrix[7], crosstalk_matrix[8],
        has_profile,
        d_in_trc0, in_trc_size_0,
        d_in_trc1, in_trc_size_1,
        d_in_trc2, in_trc_size_2,
        d_out_trc0, out_trc_size_0,
        d_out_trc1, out_trc_size_1,
        d_out_trc2, out_trc_size_2,
        matrix_3x3 ? matrix_3x3[0] : 1.0f, matrix_3x3 ? matrix_3x3[1] : 0.0f, matrix_3x3 ? matrix_3x3[2] : 0.0f,
        matrix_3x3 ? matrix_3x3[3] : 0.0f, matrix_3x3 ? matrix_3x3[4] : 1.0f, matrix_3x3 ? matrix_3x3[5] : 0.0f,
        matrix_3x3 ? matrix_3x3[6] : 0.0f, matrix_3x3 ? matrix_3x3[7] : 0.0f, matrix_3x3 ? matrix_3x3[8] : 1.0f,
        offset_3 ? offset_3[0] : 0.0f, offset_3 ? offset_3[1] : 0.0f, offset_3 ? offset_3[2] : 0.0f,
        d_clut, clut_dim_r, clut_dim_g, clut_dim_b,
        bradford_matrix[0], bradford_matrix[1], bradford_matrix[2],
        bradford_matrix[3], bradford_matrix[4], bradford_matrix[5],
        bradford_matrix[6], bradford_matrix[7], bradford_matrix[8],
        xyz_to_srgb_matrix[0], xyz_to_srgb_matrix[1], xyz_to_srgb_matrix[2],
        xyz_to_srgb_matrix[3], xyz_to_srgb_matrix[4], xyz_to_srgb_matrix[5],
        xyz_to_srgb_matrix[6], xyz_to_srgb_matrix[7], xyz_to_srgb_matrix[8],
        colorspace_type
    );

    cudaError_t err = cudaDeviceSynchronize();
    bool success = (err == cudaSuccess);

    if (success) {
        cudaMemcpy(host_output_pixels, d_output, img_size, cudaMemcpyDeviceToHost);
    } else {
        std::cerr << "CUDA Kernel failed: " << cudaGetErrorString(err) << std::endl;
    }

    if (d_output) cudaFree(d_output);
    if (d_in_trc0) cudaFree(d_in_trc0);
    if (d_in_trc1) cudaFree(d_in_trc1);
    if (d_in_trc2) cudaFree(d_in_trc2);
    if (d_out_trc0) cudaFree(d_out_trc0);
    if (d_out_trc1) cudaFree(d_out_trc1);
    if (d_out_trc2) cudaFree(d_out_trc2);
    if (d_clut) cudaFree(d_clut);

    return success;
}
