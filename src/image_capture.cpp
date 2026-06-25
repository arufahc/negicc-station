#include "image_capture.h"
#include "sony_camera_session.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <netinet/in.h>
#include "libraw/tiff_head.h"
#include <lcms2.h>
#include "lcms2_plugin.h"
#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"
#include "dcraw/gamma_curve.h"
#include "color_conversion.h"
#include <mutex>
#include <set>

static std::mutex g_temp_files_mutex;
static std::set<std::string> g_active_temp_files;

void register_temp_file(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(g_temp_files_mutex);
    g_active_temp_files.insert(filepath);
}

void unregister_temp_file(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(g_temp_files_mutex);
    g_active_temp_files.erase(filepath);
}

bool is_registered_temp_file(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(g_temp_files_mutex);
    return g_active_temp_files.find(filepath) != g_active_temp_files.end();
}

void cleanup_active_temp_files() {
    std::lock_guard<std::mutex> lock(g_temp_files_mutex);
    for (const auto& fp : g_active_temp_files) {
        std::remove(fp.c_str());
    }
    g_active_temp_files.clear();
}

// Global static object to automatically clean up remaining files on module exit / unload
struct TempFileRegistryDestructor {
    ~TempFileRegistryDestructor() {
        cleanup_active_temp_files();
    }
};
static TempFileRegistryDestructor g_temp_file_registry_destructor;

static const CapturedImage* g_cached_image_ptr = nullptr;
static std::vector<std::string> g_cached_filepaths;
static bool g_cached_half = false;
static std::vector<uint16_t> g_cached_host_raw_buf;
static int g_cached_w = 0;
static int g_cached_h = 0;

void clear_global_cache() {
    g_cached_image_ptr = nullptr;
    g_cached_filepaths.clear();
    g_cached_half = false;
    g_cached_host_raw_buf.clear();
    g_cached_w = 0;
    g_cached_h = 0;
    clear_cuda_device_cache();
}

CapturedImage::~CapturedImage() {
    if (g_cached_image_ptr == this || g_cached_filepaths == m_filepaths) {
        clear_global_cache();
    }
    for (const auto& fp : m_filepaths) {
        if (is_registered_temp_file(fp)) {
            std::remove(fp.c_str());
            unregister_temp_file(fp);
        }
    }
}

void CapturedImage::discard() {
    if (g_cached_image_ptr == this || g_cached_filepaths == m_filepaths) {
        clear_global_cache();
    }
    for (const auto& fp : m_filepaths) {
        if (is_registered_temp_file(fp)) {
            std::remove(fp.c_str());
            unregister_temp_file(fp);
        }
    }
    m_filepaths.clear();
}

static bool ensure_decoded_raw(const CapturedImage& img, bool half_size, int& out_w, int& out_h, std::vector<uint16_t>& out_raw_buf) {
    if (g_cached_image_ptr == &img && g_cached_filepaths == img.filepaths() && g_cached_half == half_size && !g_cached_host_raw_buf.empty()) {
        out_w = g_cached_w;
        out_h = g_cached_h;
        out_raw_buf = g_cached_host_raw_buf;
        return true;
    }

    clear_global_cache();

    if (img.filepaths().empty()) {
        std::cerr << "ERROR: No filepaths available in CapturedImage." << std::endl;
        return false;
    }

    bool success = false;
    if (img.capture_type() == ImageCaptureType::SINGLE) {
        std::cout << "[CapturedImage] Loading single frame to linear buffer..." << std::endl;
        LibRaw* proc = RawProcessor::load_raw(img.filepaths()[0], /*debayer*/ true, /*half_size*/ half_size, /*qual*/ 0, /*crop*/ false);
        if (!proc) {
            std::cerr << "ERROR: Failed to load single raw frame." << std::endl;
            return false;
        }

        out_w = proc->imgdata.sizes.iwidth;
        out_h = proc->imgdata.sizes.iheight;
        out_raw_buf.resize(out_w * out_h * 3);

        for (int i = 0; i < out_w * out_h; ++i) {
            out_raw_buf[i * 3]     = proc->imgdata.image[i][0];
            out_raw_buf[i * 3 + 1] = proc->imgdata.image[i][1];
            out_raw_buf[i * 3 + 2] = proc->imgdata.image[i][2];
        }

        proc->recycle();
        delete proc;
        success = true;

    } else if (img.capture_type() == ImageCaptureType::SONY_PIXEL_SHIFT_4) {
        if (img.filepaths().size() < 4) {
            std::cerr << "ERROR: Sony 4-shot pixel shift requires 4 frames, but only got " << img.filepaths().size() << std::endl;
            return false;
        }
        std::cout << "[CapturedImage] Loading and merging 4 pixel-shift frames..." << std::endl;

        LibRaw* procs[4];
        for (int i = 0; i < 4; ++i) {
            procs[i] = RawProcessor::load_raw(img.filepaths()[i], /*debayer*/ false, /*half_size*/ false, /*qual*/ 0, /*crop*/ false);
            if (!procs[i]) {
                std::cerr << "ERROR: Failed to load raw frame " << i << std::endl;
                for (int j = 0; j < i; ++j) {
                    procs[j]->recycle();
                    delete procs[j];
                }
                return false;
            }
        }

        LibRaw* proc = RawProcessor::merge_pixel_shift_raw(procs);
        if (!proc) {
            std::cerr << "ERROR: Failed to merge pixel shift frames." << std::endl;
            return false;
        }

        int w = proc->imgdata.sizes.iwidth;
        int h = proc->imgdata.sizes.iheight;

        if (half_size) {
            out_w = w / 2;
            out_h = h / 2;
            out_raw_buf.resize(out_w * out_h * 3);

            for (int row = 0; row < out_h; ++row) {
                for (int col = 0; col < out_w; ++col) {
                    long r_sum = 0, g_sum = 0, b_sum = 0;
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            int src_idx = (row * 2 + dy) * w + (col * 2 + dx);
                            r_sum += proc->imgdata.image[src_idx][0];
                            g_sum += proc->imgdata.image[src_idx][1];
                            b_sum += proc->imgdata.image[src_idx][2];
                        }
                    }
                    int dest_idx = row * out_w + col;
                    out_raw_buf[dest_idx * 3]     = static_cast<uint16_t>(r_sum / 4);
                    out_raw_buf[dest_idx * 3 + 1] = static_cast<uint16_t>(g_sum / 4);
                    out_raw_buf[dest_idx * 3 + 2] = static_cast<uint16_t>(b_sum / 4);
                }
            }
        } else {
            out_w = w;
            out_h = h;
            out_raw_buf.resize(out_w * out_h * 3);
            for (int i = 0; i < out_w * out_h; ++i) {
                out_raw_buf[i * 3]     = proc->imgdata.image[i][0];
                out_raw_buf[i * 3 + 1] = proc->imgdata.image[i][1];
                out_raw_buf[i * 3 + 2] = proc->imgdata.image[i][2];
            }
        }

        proc->recycle();
        delete proc;
        success = true;
    }

    if (success) {
        g_cached_image_ptr = &img;
        g_cached_filepaths = img.filepaths();
        g_cached_half = half_size;
        g_cached_host_raw_buf = out_raw_buf;
        g_cached_w = out_w;
        g_cached_h = out_h;
    }

    return success;
}

static int read_profile_from_file(const std::string& prof_name, unsigned **prof_out, unsigned *size);
static bool apply_lcms_profile(std::vector<uint16_t>& buf, int width, int height,
                              const std::string& input_prof_path,
                              const std::string& output_prof_path,
                              const uint8_t* input_prof_data,
                              size_t input_prof_data_size);

static inline void apply_crosstalk_correction(uint16_t& r, uint16_t& g, uint16_t& b, const std::vector<float>& cc_matrix) {
    if (cc_matrix.empty()) return;
    float fr = r * cc_matrix[0] + g * cc_matrix[1] + b * cc_matrix[2] + 0.5f;
    float fg = r * cc_matrix[3] + g * cc_matrix[4] + b * cc_matrix[5] + 0.5f;
    float fb = r * cc_matrix[6] + g * cc_matrix[7] + b * cc_matrix[8] + 0.5f;
    r = std::min(65535, (int)std::max(0.0f, fr));
    g = std::min(65535, (int)std::max(0.0f, fg));
    b = std::min(65535, (int)std::max(0.0f, fb));
}

static inline float dot_product(const std::vector<float>& v1, const std::vector<int>& v2) {
    float prod = 0;
    for (size_t i = 0; i < v1.size() && i < v2.size(); ++i) {
        prod += v1[i] * v2[i];
    }
    return prod;
}

static inline void scale_vector(std::vector<float>& v, float factor) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] *= factor;
    }
}

static void adjust_correction_matrix(std::vector<float>& r_coef, 
                                     std::vector<float>& g_coef,
                                     std::vector<float>& b_coef,
                                     float global_exposure_comp,
                                     const std::vector<int>& profile_film_base_rgb,
                                     const std::vector<int>& film_base_rgb) {
    float cc_average_r = dot_product(r_coef, film_base_rgb);
    float cc_average_g = dot_product(g_coef, film_base_rgb);
    float cc_average_b = dot_product(b_coef, film_base_rgb);
    float cc_profile_r = dot_product(r_coef, profile_film_base_rgb);
    float cc_profile_g = dot_product(g_coef, profile_film_base_rgb);
    float cc_profile_b = dot_product(b_coef, profile_film_base_rgb);

    float g_scale = 1.0f;
    float b_scale = 1.0f;

    scale_vector(r_coef, global_exposure_comp);
    scale_vector(g_coef, g_scale * global_exposure_comp);
    scale_vector(b_coef, b_scale * global_exposure_comp);
}

// Private cmsStage structs are now imported via lcms2_plugin.h

/*
 * Color conversion pipelines design notes:
 * 
 * 1. "cuda" pipeline (default):
 *    - Extracts input TRC, 3x3 matrix/offset, 3D cLUT, and output TRC manually from the film profile.
 *    - Applies them manually on the GPU, scaling to D50 XYZ PCS.
 *    - Assumes the output space is sRGB (or linear sRGB-g10), applying hardcoded Bradford adaptation (D50->D65),
 *      standard XYZ-to-sRGB matrix projection, and standard sRGB/linear EOTF curves.
 *    - Custom output profiles are NOT supported on CUDA and will trigger a fallback to the CPU ("cpp") pipeline.
 *    - For write_tiff, the output profile is only attached to the metadata and is NOT used for conversion.
 * 
 * 2. "python" pipeline:
 *    - Resides in `color_conversion.py` and operates similarly to the CUDA pipeline for sRGB targets,
 *      manually applying TRCs and CLUT, and using hardcoded sRGB conversions.
 *    - Note that calling `to_numpy` or `write_tiff` from Python with `pipeline="python"` will fall back
 *      to the C++ CPU ("cpp") pipeline, as C++ cannot execute Python code.
 * 
 * 3. "cpp" pipeline:
 *    - Uses Little CMS to create a real color transform from the film profile (IT8) to the target output profile.
 *    - This is the only pipeline that actually utilizes the custom output profile file to perform pixel color conversion.
 */
static bool run_color_pipeline_host(
    std::vector<uint16_t>& buf, int out_w, int out_h,
    const std::vector<float>& cc_matrix, float exposure_comp,
    const std::string& pipeline,
    const std::string& it8_profile_path, const std::string& output_profile_path,
    const uint8_t* it8_profile_data, size_t it8_profile_data_size
) {
    bool has_profile = !it8_profile_path.empty() || (it8_profile_data != nullptr && it8_profile_data_size > 0);
    bool use_cuda = (pipeline == "cuda" && has_profile);

    if (use_cuda && (output_profile_path == "srgb" || output_profile_path == "srgb-g10")) {
        if (is_cuda_available()) {
            int has_prof_flag = 0;
            std::vector<float> in_trc0, in_trc1, in_trc2;
            std::vector<float> out_trc0, out_trc1, out_trc2;
            std::vector<float> matrix_3x3;
            std::vector<float> offset_3;
            std::vector<float> clut_grid;
            int clut_dim_r = 0, clut_dim_g = 0, clut_dim_b = 0;

            cmsHPROFILE h_profile = nullptr;
            if (it8_profile_data && it8_profile_data_size > 0) {
                h_profile = cmsOpenProfileFromMem(it8_profile_data, it8_profile_data_size);
            } else if (!it8_profile_path.empty()) {
                unsigned* prof = nullptr;
                unsigned size = 0;
                if (read_profile_from_file(it8_profile_path, &prof, &size) == 0) {
                    h_profile = cmsOpenProfileFromMem(prof, size);
                    free(prof);
                }
            }

            if (h_profile) {
                cmsPipeline* h_pipeline = (cmsPipeline*)cmsReadTag(h_profile, cmsSigAToB0Tag);
                if (h_pipeline) {
                    has_prof_flag = 1;
                    int curve_set_count = 0;
                    cmsStage* stage_ptr = cmsPipelineGetPtrToFirstStage(h_pipeline);
                    while (stage_ptr) {
                        cmsStageSignature stage_type = cmsStageType(stage_ptr);
                        if (stage_type == cmsSigCurveSetElemType) {
                            _cmsStageToneCurvesData* curve_data = (_cmsStageToneCurvesData*)cmsStageData(stage_ptr);
                            if (curve_data->nCurves >= 3) {
                                std::vector<float>* p_trc0 = nullptr;
                                std::vector<float>* p_trc1 = nullptr;
                                std::vector<float>* p_trc2 = nullptr;
                                if (curve_set_count == 0) {
                                    p_trc0 = &in_trc0;
                                    p_trc1 = &in_trc1;
                                    p_trc2 = &in_trc2;
                                } else {
                                    p_trc0 = &out_trc0;
                                    p_trc1 = &out_trc1;
                                    p_trc2 = &out_trc2;
                                }

                                cmsToneCurve* tc0 = curve_data->TheCurves[0];
                                int entries0 = cmsGetToneCurveEstimatedTableEntries(tc0);
                                cmsUInt16Number* table0 = (cmsUInt16Number*)cmsGetToneCurveEstimatedTable(tc0);
                                p_trc0->resize(entries0);
                                for (int i = 0; i < entries0; ++i) (*p_trc0)[i] = table0[i] / 65535.0f;

                                cmsToneCurve* tc1 = curve_data->TheCurves[1];
                                int entries1 = cmsGetToneCurveEstimatedTableEntries(tc1);
                                cmsUInt16Number* table1 = (cmsUInt16Number*)cmsGetToneCurveEstimatedTable(tc1);
                                p_trc1->resize(entries1);
                                for (int i = 0; i < entries1; ++i) (*p_trc1)[i] = table1[i] / 65535.0f;

                                cmsToneCurve* tc2 = curve_data->TheCurves[2];
                                int entries2 = cmsGetToneCurveEstimatedTableEntries(tc2);
                                cmsUInt16Number* table2 = (cmsUInt16Number*)cmsGetToneCurveEstimatedTable(tc2);
                                p_trc2->resize(entries2);
                                for (int i = 0; i < entries2; ++i) (*p_trc2)[i] = table2[i] / 65535.0f;
                            }
                            curve_set_count++;
                        } else if (stage_type == cmsSigMatrixElemType) {
                            _cmsStageMatrixData* matrix_data = (_cmsStageMatrixData*)cmsStageData(stage_ptr);
                            matrix_3x3.resize(9);
                            offset_3.resize(3);
                            for (int i = 0; i < 9; ++i) matrix_3x3[i] = (float)matrix_data->Double[i];
                            for (int i = 0; i < 3; ++i) offset_3[i] = (float)matrix_data->Offset[i];
                        } else if (stage_type == cmsSigCLutElemType) {
                            _cmsStageCLutData* clut_data = (_cmsStageCLutData*)cmsStageData(stage_ptr);
                            clut_dim_r = clut_data->Params->nSamples[0];
                            clut_dim_g = clut_data->Params->nSamples[1];
                            clut_dim_b = clut_data->Params->nSamples[2];
                            int total_elements = clut_dim_r * clut_dim_g * clut_dim_b * 3;
                            clut_grid.resize(total_elements);
                            if (clut_data->HasFloatValues) {
                                float* table = clut_data->Tab.TFloat;
                                for (int i = 0; i < total_elements; ++i) clut_grid[i] = table[i];
                            } else {
                                cmsUInt16Number* table = clut_data->Tab.T;
                                for (int i = 0; i < total_elements; ++i) clut_grid[i] = table[i] / 65535.0f;
                            }
                        }
                        stage_ptr = cmsStageNext(stage_ptr);
                    }
                }
                cmsCloseProfile(h_profile);
            }

            float bradford_matrix[9] = {
                 0.9555766f, -0.0230393f,  0.0631636f,
                -0.0282895f,  1.0099416f,  0.0210077f,
                 0.0122982f, -0.0204830f,  1.3299098f
            };
            float xyz_to_srgb_matrix[9] = {
                 3.2406255f, -1.5372080f, -0.4986286f,
                -0.9689307f,  1.8757561f,  0.0415175f,
                 0.0557101f, -0.2040211f,  1.0569959f
            };
            int colorspace_type = (output_profile_path == "srgb") ? 0 : 1;
            
            std::vector<float> default_cc = {1,0,0,0,1,0,0,0,1};
            const float* cc_ptr = cc_matrix.empty() ? default_cc.data() : cc_matrix.data();

            bool cuda_ok = false;
            if (std::getenv("FORCE_CUDA_FALLBACK") != nullptr) {
                std::cerr << "FORCE_CUDA_FALLBACK environment variable detected. Bypassing CUDA pipeline run to simulate fallback." << std::endl;
            } else {
                cuda_ok = run_cuda_color_pipeline(
                    buf.data(), buf.data(), out_w, out_h,
                    cc_ptr, exposure_comp,
                    has_prof_flag,
                    in_trc0.empty() ? nullptr : in_trc0.data(), in_trc0.size(),
                    in_trc1.empty() ? nullptr : in_trc1.data(), in_trc1.size(),
                    in_trc2.empty() ? nullptr : in_trc2.data(), in_trc2.size(),
                    out_trc0.empty() ? nullptr : out_trc0.data(), out_trc0.size(),
                    out_trc1.empty() ? nullptr : out_trc1.data(), out_trc1.size(),
                    out_trc2.empty() ? nullptr : out_trc2.data(), out_trc2.size(),
                    matrix_3x3.empty() ? nullptr : matrix_3x3.data(),
                    offset_3.empty() ? nullptr : offset_3.data(),
                    clut_grid.empty() ? nullptr : clut_grid.data(),
                    clut_dim_r, clut_dim_g, clut_dim_b,
                    bradford_matrix,
                    xyz_to_srgb_matrix,
                    colorspace_type
                );
            }
            if (cuda_ok) {
                return true;
            }
            std::cerr << "WARNING: CUDA color pipeline failed. Falling back to CPU." << std::endl;
            use_cuda = false;
        } else {
            std::cerr << "WARNING: CUDA requested but not available. Falling back to CPU." << std::endl;
            use_cuda = false;
        }
    } else if (use_cuda) {
        std::cerr << "WARNING: CUDA color pipeline only supports srgb/srgb-g10. Falling back to CPU." << std::endl;
        use_cuda = false;
    }

    // Fallback: If pipeline was cuda but we ended up on CPU, the host-side crosstalk/base scaling matrix
    // was bypassed. We must apply it here on the CPU before evaluating the profile transform.
    if (pipeline == "cuda" && has_profile && !use_cuda) {
        if (!cc_matrix.empty()) {
            for (int i = 0; i < out_w * out_h; ++i) {
                uint16_t r = buf[i * 3];
                uint16_t g = buf[i * 3 + 1];
                uint16_t b = buf[i * 3 + 2];
                apply_crosstalk_correction(r, g, b, cc_matrix);
                buf[i * 3]     = r;
                buf[i * 3 + 1] = g;
                buf[i * 3 + 2] = b;
            }
        }
    }

    if (has_profile) {
        if (!apply_lcms_profile(buf, out_w, out_h, it8_profile_path, output_profile_path,
                                it8_profile_data, it8_profile_data_size)) {
            std::cerr << "WARNING: Failed to apply LCMS profile on CPU." << std::endl;
            return false;
        }
    }
    return true;
}

static int read_profile_from_file(const std::string& prof_name, unsigned **prof_out, unsigned *size) {
    FILE *fp = fopen(prof_name.c_str(), "rb");
    if (fp) {
        if (fread(size, 4, 1, fp) != 1) {
            fclose(fp);
            return -1;
        }
        fseek(fp, 0, SEEK_SET);
        *size = ntohl(*size);
        *prof_out = (unsigned *)malloc(*size);
        if (!*prof_out) {
            fclose(fp);
            return -1;
        }
        if (fread(*prof_out, 1, *size, fp) != *size) {
            free(*prof_out);
            *prof_out = nullptr;
            fclose(fp);
            return -1;
        }
        fclose(fp);
        return 0;
    }
    std::cerr << "ERROR: Cannot read ICC profile: " << prof_name << std::endl;
    return -1;
}

static bool apply_lcms_profile(std::vector<uint16_t>& buf, int width, int height,
                              const std::string& input_prof_path,
                              const std::string& output_prof_path,
                              const uint8_t* input_prof_data = nullptr,
                              size_t input_prof_data_size = 0) {
    cmsHPROFILE in_profile = nullptr;
    cmsHPROFILE out_profile = nullptr;
    cmsHTRANSFORM transform = nullptr;
    unsigned* oprof = nullptr;
    unsigned size = 0;

    // Load input (film) ICC profile: prefer in-memory buffer, fall back to file path
    if (input_prof_data && input_prof_data_size > 0) {
        in_profile = cmsOpenProfileFromMem(input_prof_data, (cmsUInt32Number)input_prof_data_size);
        if (!in_profile) {
            std::cerr << "ERROR: Failed to open input ICC profile from memory buffer." << std::endl;
            return false;
        }
    } else {
        unsigned* prof = nullptr;
        if (read_profile_from_file(input_prof_path, &prof, &size) == 0) {
            in_profile = cmsOpenProfileFromMem(prof, size);
            free(prof);
        }
        if (!in_profile) {
            std::cerr << "ERROR: Failed to open input ICC profile from " << input_prof_path << std::endl;
            return false;
        }
    }

    if (output_prof_path == "srgb") {
        out_profile = cmsOpenProfileFromMem(sRGB_elle_V2_srgbtrc_icc, sRGB_elle_V2_srgbtrc_icc_len);
    } else if (output_prof_path == "srgb-g10") {
        out_profile = cmsOpenProfileFromMem(sRGB_elle_V2_g10_icc, sRGB_elle_V2_g10_icc_len);
    } else if (read_profile_from_file(output_prof_path, &oprof, &size) == 0) {
        out_profile = cmsOpenProfileFromMem(oprof, size);
        free(oprof);
    }

    if (!out_profile) {
        std::cerr << "ERROR: Failed to open output ICC profile from " << output_prof_path << std::endl;
        cmsCloseProfile(in_profile);
        return false;
    }

    transform = cmsCreateTransform(in_profile, TYPE_RGB_16, out_profile, TYPE_RGB_16, INTENT_PERCEPTUAL, 0);
    if (!transform) {
        std::cerr << "ERROR: Failed to create LCMS transform." << std::endl;
        cmsCloseProfile(out_profile);
        cmsCloseProfile(in_profile);
        return false;
    }

    cmsDoTransform(transform, buf.data(), buf.data(), width * height);

    cmsDeleteTransform(transform);
    cmsCloseProfile(out_profile);
    cmsCloseProfile(in_profile);
    return true;
}

bool CapturedImage::get_linear_rgb(bool half_size, int& out_w, int& out_h, std::vector<uint16_t>& out_buf,
                                   const std::vector<float>& cc_matrix,
                                   const std::string& it8_profile_path,
                                   const std::string& output_profile_path,
                                   const std::vector<int>& profile_film_base,
                                   const std::vector<int>& film_base,
                                   float exposure_comp,
                                   const std::string& pipeline,
                                   const uint8_t* it8_profile_data,
                                   size_t it8_profile_data_size) const {
    if (m_filepaths.empty()) {
        std::cerr << "ERROR: No filepaths available in CapturedImage." << std::endl;
        return false;
    }

    std::vector<float> adjusted_cc = cc_matrix;
    bool has_profile = !it8_profile_path.empty() || (it8_profile_data != nullptr && it8_profile_data_size > 0);
    if (has_profile) {
        if (adjusted_cc.empty()) {
            adjusted_cc = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        }
        std::vector<int> p_fb = profile_film_base;
        std::vector<int> c_fb = film_base;
        if (p_fb.empty() || c_fb.empty()) {
            p_fb = {1, 1, 1};
            c_fb = {1, 1, 1};
        }
        std::vector<float> r_coef = {adjusted_cc[0], adjusted_cc[1], adjusted_cc[2]};
        std::vector<float> g_coef = {adjusted_cc[3], adjusted_cc[4], adjusted_cc[5]};
        std::vector<float> b_coef = {adjusted_cc[6], adjusted_cc[7], adjusted_cc[8]};
        adjust_correction_matrix(r_coef, g_coef, b_coef, exposure_comp, p_fb, c_fb);
        adjusted_cc = {
            r_coef[0], r_coef[1], r_coef[2],
            g_coef[0], g_coef[1], g_coef[2],
            b_coef[0], b_coef[1], b_coef[2]
        };
    }

    std::vector<float> cpu_cc = adjusted_cc;
    if (pipeline == "cuda" && has_profile) {
        cpu_cc.clear(); // Bypassed on CPU, applied on GPU
    }

    if (!ensure_decoded_raw(*this, half_size, out_w, out_h, out_buf)) {
        return false;
    }

    if (!cpu_cc.empty()) {
        for (int i = 0; i < out_w * out_h; ++i) {
            uint16_t r = out_buf[i * 3];
            uint16_t g = out_buf[i * 3 + 1];
            uint16_t b = out_buf[i * 3 + 2];
            apply_crosstalk_correction(r, g, b, cpu_cc);
            out_buf[i * 3]     = r;
            out_buf[i * 3 + 1] = g;
            out_buf[i * 3 + 2] = b;
        }
    }

    if (!run_color_pipeline_host(out_buf, out_w, out_h, adjusted_cc, exposure_comp,
                                 pipeline, it8_profile_path, output_profile_path,
                                 it8_profile_data, it8_profile_data_size)) {
        return false;
    }

    return true;
}

std::unique_ptr<CapturedImage> capture_image(ImageCaptureType type, uint32_t shutterSpeedVal) {
    std::unique_ptr<CameraSession> session = std::make_unique<SonyCameraSession>();
    if (!session->initialize()) {
        std::cerr << "ERROR: Failed to initialize camera session." << std::endl;
        return nullptr;
    }
    if (!session->configure_settings()) {
        std::cerr << "ERROR: Failed to configure settings." << std::endl;
        session->close();
        return nullptr;
    }
    if (!session->set_shutter_speed(shutterSpeedVal)) {
        std::cerr << "ERROR: Failed to set shutter speed." << std::endl;
        session->close();
        return nullptr;
    }

    CaptureOutput output;
    CaptureType capType = (type == ImageCaptureType::SONY_PIXEL_SHIFT_4)
                          ? CaptureType::SONY_PIXEL_SHIFT_4
                          : CaptureType::SINGLE;

    if (!session->capture(capType, output)) {
        std::cerr << "ERROR: Capture failed." << std::endl;
        session->close();
        return nullptr;
    }
    session->close();

    uint16_t numerator = shutterSpeedVal >> 16;
    uint16_t denominator = shutterSpeedVal & 0xFFFF;
    double shutterSec = (denominator > 0) ? (double)numerator / (double)denominator : 0.1;

    return std::make_unique<CapturedImage>(type, shutterSec, 100, output.filepaths);
}

bool write_linear_tiff(const CapturedImage& img,
                       const std::string& output_path,
                       bool half_size,
                       const std::vector<float>& cc_matrix,
                       const std::string& it8_profile_path,
                       const std::string& output_profile_path,
                       const std::vector<int>& profile_film_base,
                       const std::vector<int>& film_base,
                       float exposure_comp,
                       const std::string& pipeline,
                       const uint8_t* it8_profile_data,
                       size_t it8_profile_data_size) {
    if (img.filepaths().empty()) return false;

    int out_w = 0, out_h = 0;
    std::vector<uint16_t> buf;
    if (!ensure_decoded_raw(img, half_size, out_w, out_h, buf)) {
        return false;
    }

    std::vector<float> adjusted_cc = cc_matrix;
    bool has_profile = !it8_profile_path.empty() || (it8_profile_data != nullptr && it8_profile_data_size > 0);
    if (has_profile) {
        if (adjusted_cc.empty()) {
            adjusted_cc = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        }
        std::vector<int> p_fb = profile_film_base;
        std::vector<int> c_fb = film_base;
        if (p_fb.empty() || c_fb.empty()) {
            p_fb = {1, 1, 1};
            c_fb = {1, 1, 1};
        }
        std::vector<float> r_coef = {adjusted_cc[0], adjusted_cc[1], adjusted_cc[2]};
        std::vector<float> g_coef = {adjusted_cc[3], adjusted_cc[4], adjusted_cc[5]};
        std::vector<float> b_coef = {adjusted_cc[6], adjusted_cc[7], adjusted_cc[8]};
        adjust_correction_matrix(r_coef, g_coef, b_coef, exposure_comp, p_fb, c_fb);
        adjusted_cc = {
            r_coef[0], r_coef[1], r_coef[2],
            g_coef[0], g_coef[1], g_coef[2],
            b_coef[0], b_coef[1], b_coef[2]
        };
    }

    std::vector<float> cpu_cc = adjusted_cc;
    if (pipeline == "cuda" && has_profile) {
        cpu_cc.clear();
    }

    if (!cpu_cc.empty()) {
        for (int i = 0; i < out_w * out_h; ++i) {
            uint16_t r = buf[i * 3];
            uint16_t g = buf[i * 3 + 1];
            uint16_t b = buf[i * 3 + 2];
            apply_crosstalk_correction(r, g, b, cpu_cc);
            buf[i * 3]     = r;
            buf[i * 3 + 1] = g;
            buf[i * 3 + 2] = b;
        }
    }

    if (!run_color_pipeline_host(buf, out_w, out_h, adjusted_cc, exposure_comp,
                                 pipeline, it8_profile_path, output_profile_path,
                                 it8_profile_data, it8_profile_data_size)) {
        std::cerr << "WARNING: Failed to run color pipeline." << std::endl;
        return false;
    }

    const uint8_t* output_profile_bytes = nullptr;
    size_t profile_size = 0;
    std::vector<uint8_t> custom_profile_buf;

    if (output_profile_path == "srgb") {
        output_profile_bytes = sRGB_elle_V2_srgbtrc_icc;
        profile_size = sRGB_elle_V2_srgbtrc_icc_len;
    } else if (output_profile_path == "srgb-g10") {
        output_profile_bytes = sRGB_elle_V2_g10_icc;
        profile_size = sRGB_elle_V2_g10_icc_len;
    } else if (!output_profile_path.empty()) {
        unsigned* prof_ptr = nullptr;
        unsigned read_size = 0;
        if (read_profile_from_file(output_profile_path, &prof_ptr, &read_size) == 0) {
            custom_profile_buf.assign(reinterpret_cast<uint8_t*>(prof_ptr), reinterpret_cast<uint8_t*>(prof_ptr) + read_size);
            free(prof_ptr);
            output_profile_bytes = custom_profile_buf.data();
            profile_size = read_size;
        }
    }

    LibRaw metadata_proc;
    if (metadata_proc.open_file(img.filepaths()[0].c_str()) != LIBRAW_SUCCESS) {
        std::cerr << "ERROR: Failed to open raw file for metadata." << std::endl;
        return false;
    }
    metadata_proc.imgdata.idata.colors = 3;
    metadata_proc.imgdata.params.output_bps = 16;

    struct tiff_hdr header;
    tiff_head(&metadata_proc, &header, profile_size, out_w, out_h);

    FILE* fp = fopen(output_path.c_str(), "wb+");
    if (!fp) {
        metadata_proc.recycle();
        return false;
    }

    fwrite(&header, sizeof(header), 1, fp);

    if (profile_size && output_profile_bytes) {
        fwrite(output_profile_bytes, 1, profile_size, fp);
    }

    fwrite(buf.data(), 2 * 3, out_w * out_h, fp);

    fclose(fp);
    metadata_proc.recycle();
    return true;
}

std::string CapturedImage::camera_model() const {
    if (m_filepaths.empty()) return "Unknown";
    LibRaw proc;
    if (proc.open_file(m_filepaths[0].c_str()) == LIBRAW_SUCCESS) {
        std::string model = proc.imgdata.idata.model;
        proc.recycle();
        return model;
    }
    return "Unknown";
}
