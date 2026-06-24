#ifndef IMAGE_CAPTURE_H
#define IMAGE_CAPTURE_H

#include <string>
#include <vector>
#include <memory>
#include "camera_session.h"
#include "raw_processor.h"

enum class ImageCaptureType {
    SINGLE = 0,
    SONY_PIXEL_SHIFT_4 = 1
};

class CapturedImage {
public:
    CapturedImage(ImageCaptureType type, double shutter_speed, int iso, const std::vector<std::string>& filepaths)
        : m_type(type), m_shutter_speed(shutter_speed), m_iso(iso), m_filepaths(filepaths) {}
    ~CapturedImage();

    ImageCaptureType capture_type() const { return m_type; }
    double shutter_speed() const { return m_shutter_speed; }
    int iso() const { return m_iso; }
    const std::vector<std::string>& filepaths() const { return m_filepaths; }
    std::string camera_model() const;

    // Deletes the temporary raw files from disk
    void discard();

    // Decodes the raw files and fills out_buf with linear RGB 16-bit values.
    // Dimensions out_w and out_h are set.
    // it8_profile_path: path to ICC file on disk (legacy). If empty and it8_profile_data is non-null,
    // the ICC profile is loaded directly from the provided memory buffer.
    bool get_linear_rgb(bool half_size, int& out_w, int& out_h, std::vector<uint16_t>& out_buf,
                        const std::vector<float>& cc_matrix = {},
                        const std::string& it8_profile_path = "",
                        const std::string& output_profile_path = "srgb",
                        const std::vector<int>& profile_film_base = {},
                        const std::vector<int>& film_base = {},
                        float exposure_comp = 1.0f,
                        const std::string& pipeline = "cuda",
                        const uint8_t* it8_profile_data = nullptr,
                        size_t it8_profile_data_size = 0) const;

private:
    ImageCaptureType m_type;
    double m_shutter_speed;
    int m_iso;
    std::vector<std::string> m_filepaths;
};

// Captures a picture using the connected camera session
std::unique_ptr<CapturedImage> capture_image(ImageCaptureType type, uint32_t shutterSpeedVal);

// Stores the linear image from CapturedImage to a 16-bit RGB TIFF file
bool write_linear_tiff(const CapturedImage& img,
                       const std::string& output_path,
                       bool half_size,
                       const std::vector<float>& cc_matrix = {},
                       const std::string& it8_profile_path = "",
                       const std::string& output_profile_path = "srgb",
                       const std::vector<int>& profile_film_base = {},
                       const std::vector<int>& film_base = {},
                       float exposure_comp = 1.0f,
                       const std::string& pipeline = "cuda",
                       const uint8_t* it8_profile_data = nullptr,
                       size_t it8_profile_data_size = 0);

void clear_global_cache();

#endif // IMAGE_CAPTURE_H
