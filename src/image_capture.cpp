#include "image_capture.h"
#include "sony_camera_session.h"
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <netinet/in.h>
#include "libraw/tiff_head.h"

static inline void apply_crosstalk_correction(uint16_t& r, uint16_t& g, uint16_t& b, const std::vector<float>& cc_matrix) {
    if (cc_matrix.empty()) return;
    float fr = r * cc_matrix[0] + g * cc_matrix[1] + b * cc_matrix[2] + 0.5f;
    float fg = r * cc_matrix[3] + g * cc_matrix[4] + b * cc_matrix[5] + 0.5f;
    float fb = r * cc_matrix[6] + g * cc_matrix[7] + b * cc_matrix[8] + 0.5f;
    r = std::min(65535, (int)std::max(0.0f, fr));
    g = std::min(65535, (int)std::max(0.0f, fg));
    b = std::min(65535, (int)std::max(0.0f, fb));
}

bool CapturedImage::get_linear_rgb(bool half_size, int& out_w, int& out_h, std::vector<uint16_t>& out_buf, const std::vector<float>& cc_matrix) const {
    if (m_filepaths.empty()) {
        std::cerr << "ERROR: No filepaths available in CapturedImage." << std::endl;
        return false;
    }

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
            apply_crosstalk_correction(r, g, b, cc_matrix);
            out_buf[i * 3]     = r;
            out_buf[i * 3 + 1] = g;
            out_buf[i * 3 + 2] = b;
        }

        proc->recycle();
        delete proc;
        return true;

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
                    apply_crosstalk_correction(r, g, b, cc_matrix);
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
                apply_crosstalk_correction(r, g, b, cc_matrix);
                out_buf[i * 3]     = r;
                out_buf[i * 3 + 1] = g;
                out_buf[i * 3 + 2] = b;
            }
        }

        proc->recycle();
        delete proc;
        return true;
    }

    return false;
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
