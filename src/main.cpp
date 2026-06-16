#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <filesystem>
#include "camera_session.h"
#include "sony_camera_session.h"
#include "raw_processor.h"

/**
 * @brief Converts the captured raw output into a linear image.
 *
 * For SINGLE captures, it loads and debayers the raw image into a linear format.
 * For SONY_PIXEL_SHIFT_4 captures, it loads the 4 raw frames without individual
 * interpolation (debayer=false) and merges them using the pixel-shift algorithm.
 *
 * @param captureOutput The captured files and capture type.
 * @return A LibRaw pointer containing the processed linear image, or nullptr on failure.
 */
LibRaw* convert_to_linear(const CaptureOutput& captureOutput) {
    if (captureOutput.type == CaptureType::SINGLE) {
        if (captureOutput.filepaths.empty()) return nullptr;
        std::cout << "Converting single capture to linear..." << std::endl;
        // Debayer=true is used to produce a standard color image from the single raw frame
        return RawProcessor::load_raw(captureOutput.filepaths[0], /*debayer*/ true);
    } else if (captureOutput.type == CaptureType::SONY_PIXEL_SHIFT_4) {
        if (captureOutput.filepaths.size() < 4) {
            std::cerr << "ERROR: SONY_PIXEL_SHIFT_4 requires 4 images, but got " 
                      << captureOutput.filepaths.size() << std::endl;
            return nullptr;
        }
        LibRaw* proc[4];
        for (int i = 0; i < 4; ++i) {
            // Load frames un-interpolated so they can be merged directly
            proc[i] = RawProcessor::load_raw(captureOutput.filepaths[i], /*debayer*/ false);
            if (proc[i] == nullptr) {
                std::cerr << "ERROR: Failed to load raw image " << i << " for merge." << std::endl;
                for (int j = 0; j < i; ++j) {
                    proc[j]->recycle();
                    delete proc[j];
                }
                return nullptr;
            }
        }
        std::cout << "Merging and converting 4-shot pixel shift to linear..." << std::endl;
        return RawProcessor::merge_pixel_shift_raw(proc);
    }
    return nullptr;
}

/**
 * @brief Executes a single test case for capturing and conversion.
 */
bool run_test_case(CameraSession& session, CaptureType type, uint32_t shutterSpeedVal, double expectedShutterSec, const std::string& label) {
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "TEST CASE: " << label << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 1. Set Shutter Speed
    if (!session.set_shutter_speed(shutterSpeedVal)) {
        std::cerr << "ERROR: Failed to set shutter speed." << std::endl;
        return false;
    }

    // 2. Capture
    CaptureOutput output;
    if (!session.capture(type, output)) {
        std::cerr << "ERROR: Capture failed." << std::endl;
        return false;
    }

    // 3. Convert to Linear and measure duration
    auto start_time = std::chrono::high_resolution_clock::now();
    LibRaw* proc = convert_to_linear(output);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (proc == nullptr) {
        std::cerr << "ERROR: Failed to convert captured output to linear." << std::endl;
        return false;
    }
    std::chrono::duration<double> diff_time = end_time - start_time;
    std::cout << "Linear conversion took: " << diff_time.count() << " seconds." << std::endl;

    // 4. Verify properties
    float shutter = proc->imgdata.other.shutter;
    float iso = proc->imgdata.other.iso_speed;
    float aperture = proc->imgdata.other.aperture;

    std::cout << "\n>>> Decoded Metadata <<<" << std::endl;
    std::cout << "  Shutter Speed: " << shutter << " seconds" << std::endl;
    std::cout << "  ISO Speed:     " << iso << std::endl;
    std::cout << "  Aperture:      f/" << aperture << std::endl;

    // Clean up LibRaw processor resources
    proc->recycle();
    delete proc;

    std::cout << "\nVerifying metadata properties..." << std::endl;
    bool success = true;

    // Verify shutter speed (within 10% tolerance)
    double diff = std::abs(shutter - expectedShutterSec);
    double tolerance = expectedShutterSec * 0.10;
    if (diff > tolerance) {
        std::cerr << "VERIFICATION FAILURE: Decoded Shutter Speed (" << shutter 
                  << "s) does not match expected value (" << expectedShutterSec << "s)" << std::endl;
        success = false;
    } else {
        std::cout << "  [PASS] Shutter Speed matches expected value." << std::endl;
    }

    // Verify ISO
    if (std::abs(iso - 100.0) > 1.0) {
        std::cerr << "VERIFICATION FAILURE: Decoded ISO (" << iso 
                  << ") does not match expected base ISO (100)" << std::endl;
        success = false;
    } else {
        std::cout << "  [PASS] ISO matches expected value (100)." << std::endl;
    }


    // 5. Delete captured files
    for (const auto& filepath : output.filepaths) {
        std::cout << "Deleting captured file: " << filepath << std::endl;
        if (std::remove(filepath.c_str()) != 0) {
            std::cerr << "WARNING: Failed to delete " << filepath << std::endl;
        }
    }

    return success;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Sony A7R4 Tethered Capture Test Utility (Refactored)" << std::endl;
    std::cout << "========================================" << std::endl;

    std::unique_ptr<CameraSession> session = std::make_unique<SonyCameraSession>();

    if (!session->initialize()) {
        std::cerr << "ERROR: Failed to initialize camera session." << std::endl;
        return -1;
    }

    if (!session->configure_settings()) {
        std::cerr << "ERROR: Failed to configure settings." << std::endl;
        return -1;
    }

    bool testSuccess = true;

    // Test Case 1: SINGLE shot at 1s Exposure
    // 0x000A000A = 10/10s = 1s
    if (!run_test_case(*session, CaptureType::SINGLE, 0x000A000A, 1.0, "SINGLE (1s Exposure)")) {
        testSuccess = false;
    }

    // Test Case 2: SONY_PIXEL_SHIFT_4 shots at 1/125s Exposure
    // 0x0001007D = 1/125s
    if (!run_test_case(*session, CaptureType::SONY_PIXEL_SHIFT_4, 0x0001007D, 1.0 / 125.0, "SONY_PIXEL_SHIFT_4 (1/125s Exposure)")) {
        testSuccess = false;
    }

    // Test Case 3: SINGLE shot at 1/8s Exposure (Tests reduced dynamic timeouts)
    // 0x00010008 = 1/8s
    if (!run_test_case(*session, CaptureType::SINGLE, 0x00010008, 0.125, "SINGLE (1/8s Exposure)")) {
        testSuccess = false;
    }

    session->close();

    if (testSuccess) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL TESTS COMPLETED: SUCCESS" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "ALL TESTS COMPLETED: FAILURE" << std::endl;
        std::cout << "========================================" << std::endl;
        return -1;
    }
}
