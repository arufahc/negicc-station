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

    ImageCaptureType capture_type() const { return m_type; }
    double shutter_speed() const { return m_shutter_speed; }
    int iso() const { return m_iso; }
    const std::vector<std::string>& filepaths() const { return m_filepaths; }

    // Deletes the temporary raw files from disk
    void discard() {
        for (const auto& fp : m_filepaths) {
            std::remove(fp.c_str());
        }
        m_filepaths.clear();
    }

    // Decodes the raw files and fills out_buf with linear RGB 16-bit values.
    // Dimensions out_w and out_h are set.
    bool get_linear_rgb(bool half_size, int& out_w, int& out_h, std::vector<uint16_t>& out_buf) const;

private:
    ImageCaptureType m_type;
    double m_shutter_speed;
    int m_iso;
    std::vector<std::string> m_filepaths;
};

// Captures a picture using the connected camera session
std::unique_ptr<CapturedImage> capture_image(ImageCaptureType type, uint32_t shutterSpeedVal);

// Stores the linear image from CapturedImage to a 16-bit RGB TIFF file
bool write_linear_tiff(const CapturedImage& img, const std::string& output_path, bool half_size);

#endif // IMAGE_CAPTURE_H
