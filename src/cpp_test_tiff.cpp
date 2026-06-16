#include <iostream>
#include <vector>
#include <string>
#include "image_capture.h"
#include "libraw/libraw_internal.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <output_tiff> <half_size (0 or 1)> <raw_file_1> [raw_file_2] [raw_file_3] [raw_file_4]" << std::endl;
        return 1;
    }

    std::string output_tiff = argv[1];
    bool half_size = (std::stoi(argv[2]) != 0);

    std::vector<std::string> raw_files;
    for (int i = 3; i < argc; ++i) {
        raw_files.push_back(argv[i]);
    }

    ImageCaptureType type = (raw_files.size() == 4) ? ImageCaptureType::SONY_PIXEL_SHIFT_4 : ImageCaptureType::SINGLE;
    CapturedImage img(type, 1.0, 100, raw_files);

    std::cout << "TIFF_HEADER_SIZE:" << sizeof(struct tiff_hdr) << std::endl;

    if (!write_linear_tiff(img, output_tiff, half_size)) {
        std::cerr << "ERROR: Failed to write linear TIFF." << std::endl;
        return 1;
    }

    std::cout << "TIFF written successfully." << std::endl;
    return 0;
}
