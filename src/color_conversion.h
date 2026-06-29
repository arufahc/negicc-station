#ifndef COLOR_CONVERSION_H
#define COLOR_CONVERSION_H

#include <stdint.h>
#include <vector>

// Returns true if CUDA driver/devices are available and functional
bool is_cuda_available();

// Clears the cached CUDA device float32 raw image buffer
void clear_cuda_device_cache();

// Runs the CUDA-based color conversion pipeline
bool run_cuda_color_pipeline(
    const uint16_t* host_input_pixels,
    uint16_t* host_output_pixels,
    int w, int h,
    const float* crosstalk_matrix, // 9 floats
    float exposure_comp,
    int has_profile,
    const float* in_trc_curve_0, int in_trc_size_0,
    const float* in_trc_curve_1, int in_trc_size_1,
    const float* in_trc_curve_2, int in_trc_size_2,
    const float* out_trc_curve_0, int out_trc_size_0,
    const float* out_trc_curve_1, int out_trc_size_1,
    const float* out_trc_curve_2, int out_trc_size_2,
    const float* matrix_3x3,       // 9 floats
    const float* offset_3,         // 3 floats
    const float* clut_grid,        // flat float grid array
    int clut_dim_r, int clut_dim_g, int clut_dim_b,
    const float* bradford_matrix,  // 9 floats
    const float* xyz_to_srgb_matrix, // 9 floats
    int colorspace_type            // 0 = sRGB piecewise, 1 = sRGB-g10 linear
);

bool run_cuda_color_pipeline_uint8(
    const uint16_t* host_input_pixels,
    uint8_t* host_output_pixels,
    int w, int h,
    const float* crosstalk_matrix, // 9 floats
    float exposure_comp,
    int has_profile,
    const float* in_trc_curve_0, int in_trc_size_0,
    const float* in_trc_curve_1, int in_trc_size_1,
    const float* in_trc_curve_2, int in_trc_size_2,
    const float* out_trc_curve_0, int out_trc_size_0,
    const float* out_trc_curve_1, int out_trc_size_1,
    const float* out_trc_curve_2, int out_trc_size_2,
    const float* matrix_3x3,       // 9 floats
    const float* offset_3,         // 3 floats
    const float* clut_grid,        // flat float grid array
    int clut_dim_r, int clut_dim_g, int clut_dim_b,
    const float* bradford_matrix,  // 9 floats
    const float* xyz_to_srgb_matrix, // 9 floats
    int colorspace_type            // 0 = sRGB piecewise, 1 = sRGB-g10 linear
);

bool run_cuda_gains_histogram_search(
    const uint16_t* host_input_pixels,
    uint32_t* host_histograms,
    int w, int h,
    int x_start, int y_start, int crop_w, int crop_h,
    const float* host_cc_matrices,
    int num_configs,
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
    int clut_dim_r, int clut_dim_g, int clut_dim_b
);

#endif // COLOR_CONVERSION_H
