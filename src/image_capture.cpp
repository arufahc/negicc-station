#include "image_capture.h"
#include "sony_camera_session.h"
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <netinet/in.h>
#include "libraw/tiff_head.h"
#include <lcms2.h>
#include "elle_icc_profiles/sRGB_elle_V2_g10.h"
#include "elle_icc_profiles/sRGB_elle_V2_srgbtrc.h"
#include "dcraw/gamma_curve.h"

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

static void apply_gamma_to_buf(std::vector<uint16_t>& buf, double gamma) {
    if (std::abs(gamma - 1.0) < 1e-5) return;
    std::vector<uint16_t> curve(0x10000);
    gamma_curve(curve.data(), gamma, 0.0);
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = curve[buf[i]];
    }
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
                                   float post_correction_gamma,
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

    bool success = false;
    if (m_type == ImageCaptureType::SINGLE) {
        std::cout << "[CapturedImage] Loading single frame to linear buffer..." << std::endl;
        LibRaw* proc = RawProcessor::load_raw(m_filepaths[0], /*debayer*/ true, /*half_size*/ half_size, /*qual*/ 0, /*crop*/ false);
        if (!proc) {
            std::cerr << "ERROR: Failed to load single raw frame." << std::endl;
            return false;
        }

        out_w = proc->imgdata.sizes.iwidth;
        out_h = proc->imgdata.sizes.iheight;
        out_buf.resize(out_w * out_h * 3);

        for (int i = 0; i < out_w * out_h; ++i) {
            uint16_t r = proc->imgdata.image[i][0];
            uint16_t g = proc->imgdata.image[i][1];
            uint16_t b = proc->imgdata.image[i][2];
            apply_crosstalk_correction(r, g, b, adjusted_cc);
            out_buf[i * 3]     = r;
            out_buf[i * 3 + 1] = g;
            out_buf[i * 3 + 2] = b;
        }

        proc->recycle();
        delete proc;
        success = true;

    } else if (m_type == ImageCaptureType::SONY_PIXEL_SHIFT_4) {
        if (m_filepaths.size() < 4) {
            std::cerr << "ERROR: Sony 4-shot pixel shift requires 4 frames, but only got " << m_filepaths.size() << std::endl;
            return false;
        }
        std::cout << "[CapturedImage] Loading and merging 4 pixel-shift frames..." << std::endl;

        LibRaw* procs[4];
        for (int i = 0; i < 4; ++i) {
            procs[i] = RawProcessor::load_raw(m_filepaths[i], /*debayer*/ false, /*half_size*/ false, /*qual*/ 0, /*crop*/ false);
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
            out_buf.resize(out_w * out_h * 3);

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
                    uint16_t r = static_cast<uint16_t>(r_sum / 4);
                    uint16_t g = static_cast<uint16_t>(g_sum / 4);
                    uint16_t b = static_cast<uint16_t>(b_sum / 4);
                    apply_crosstalk_correction(r, g, b, adjusted_cc);
                    out_buf[dest_idx * 3]     = r;
                    out_buf[dest_idx * 3 + 1] = g;
                    out_buf[dest_idx * 3 + 2] = b;
                }
            }
        } else {
            out_w = w;
            out_h = h;
            out_buf.resize(out_w * out_h * 3);
            for (int i = 0; i < out_w * out_h; ++i) {
                uint16_t r = proc->imgdata.image[i][0];
                uint16_t g = proc->imgdata.image[i][1];
                uint16_t b = proc->imgdata.image[i][2];
                apply_crosstalk_correction(r, g, b, adjusted_cc);
                out_buf[i * 3]     = r;
                out_buf[i * 3 + 1] = g;
                out_buf[i * 3 + 2] = b;
            }
        }

        proc->recycle();
        delete proc;
        success = true;
    }

    if (success && has_profile) {
        apply_gamma_to_buf(out_buf, post_correction_gamma);
        if (!apply_lcms_profile(out_buf, out_w, out_h, it8_profile_path, output_profile_path,
                                it8_profile_data, it8_profile_data_size)) {
            std::cerr << "WARNING: Failed to apply LCMS profile." << std::endl;
            return false;
        }
    }

    return success;
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

bool write_linear_tiff(const CapturedImage& img, const std::string& output_path, bool half_size, const std::vector<float>& cc_matrix) {
    if (img.filepaths().empty()) return false;

    LibRaw* proc = nullptr;
    if (img.capture_type() == ImageCaptureType::SINGLE) {
        proc = RawProcessor::load_raw(img.filepaths()[0], /*debayer*/ true, /*half_size*/ half_size, /*qual*/ 0, /*crop*/ false);
    } else {
        if (img.filepaths().size() < 4) return false;
        LibRaw* procs[4];
        for (int i = 0; i < 4; ++i) {
            procs[i] = RawProcessor::load_raw(img.filepaths()[i], /*debayer*/ false, /*half_size*/ false, /*qual*/ 0, /*crop*/ false);
            if (!procs[i]) {
                for (int j = 0; j < i; ++j) {
                    procs[j]->recycle();
                    delete procs[j];
                }
                return false;
            }
        }
        proc = RawProcessor::merge_pixel_shift_raw(procs);
    }

    if (!proc) return false;

    const unsigned height = proc->imgdata.sizes.iheight;
    const unsigned width = proc->imgdata.sizes.iwidth;
    struct tiff_hdr header;
    unsigned profile_size = 0;

    if (half_size && img.capture_type() == ImageCaptureType::SINGLE) {
        tiff_head(proc, &header, profile_size);
    } else if (half_size && img.capture_type() == ImageCaptureType::SONY_PIXEL_SHIFT_4) {
        tiff_head(proc, &header, profile_size, width / 2, height / 2);
    } else {
        tiff_head(proc, &header, profile_size);
    }

    FILE* fp = fopen(output_path.c_str(), "wb+");
    if (!fp) {
        proc->recycle();
        delete proc;
        return false;
    }

    fwrite(&header, sizeof(header), 1, fp);

    if (half_size && img.capture_type() == ImageCaptureType::SONY_PIXEL_SHIFT_4) {
        const unsigned output_width = width / 2;
        std::vector<ushort> row_buf(output_width * 3);
        for (unsigned row = 0; row < height; ++row) {
            if (row % 2 == 0) {
                for (unsigned col = 0; col < output_width; ++col) {
                    row_buf[col * 3]     = (proc->imgdata.image[row * width + 2 * col][0] +
                                            proc->imgdata.image[row * width + 2 * col + 1][0]);
                    row_buf[col * 3 + 1] = (proc->imgdata.image[row * width + 2 * col][1] +
                                            proc->imgdata.image[row * width + 2 * col + 1][1]);
                    row_buf[col * 3 + 2] = (proc->imgdata.image[row * width + 2 * col][2] +
                                            proc->imgdata.image[row * width + 2 * col + 1][2]);
                }
            } else {
                for (unsigned col = 0; col < output_width; ++col) {
                    uint16_t r = (row_buf[col * 3] +
                                  proc->imgdata.image[row * width + 2 * col][0] +
                                  proc->imgdata.image[row * width + 2 * col + 1][0]) / 4;
                    uint16_t g = (row_buf[col * 3 + 1] +
                                  proc->imgdata.image[row * width + 2 * col][1] +
                                  proc->imgdata.image[row * width + 2 * col + 1][1]) / 4;
                    uint16_t b = (row_buf[col * 3 + 2] +
                                  proc->imgdata.image[row * width + 2 * col][2] +
                                  proc->imgdata.image[row * width + 2 * col + 1][2]) / 4;
                    apply_crosstalk_correction(r, g, b, cc_matrix);
                    row_buf[col * 3]     = r;
                    row_buf[col * 3 + 1] = g;
                    row_buf[col * 3 + 2] = b;
                }
                fwrite(row_buf.data(), 2 * 3, output_width, fp);
            }
        }
    } else {
        std::vector<ushort> row_buf(width * 3);
        for (unsigned row = 0; row < height; ++row) {
            for (unsigned col = 0; col < width; ++col) {
                uint16_t r = proc->imgdata.image[row * width + col][0];
                uint16_t g = proc->imgdata.image[row * width + col][1];
                uint16_t b = proc->imgdata.image[row * width + col][2];
                apply_crosstalk_correction(r, g, b, cc_matrix);
                row_buf[col * 3]     = r;
                row_buf[col * 3 + 1] = g;
                row_buf[col * 3 + 2] = b;
            }
            fwrite(row_buf.data(), 2 * 3, width, fp);
        }
    }

    fclose(fp);
    proc->recycle();
    delete proc;
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
