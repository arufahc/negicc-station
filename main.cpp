#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include "CameraRemote_SDK.h"
#include "IDeviceCallback.h"
#include "ICrCameraObjectInfo.h"
#include "raw_processor.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <dirent.h>
#include <fstream>
#include <cstring>
#include <cerrno>

bool reset_sony_camera_usb() {
    std::string devices_dir = "/sys/bus/usb/devices/";
    DIR* dir = opendir(devices_dir.c_str());
    if (!dir) {
        std::cerr << "[USB Reset] Cannot open /sys/bus/usb/devices/" << std::endl;
        return false;
    }

    bool reset_done = false;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        std::string vendor_path = devices_dir + name + "/idVendor";
        std::ifstream vendor_file(vendor_path);
        if (!vendor_file.is_open()) continue;

        std::string vendor;
        vendor_file >> vendor;
        vendor_file.close();

        if (vendor == "054c") {
            std::string bus_path = devices_dir + name + "/busnum";
            std::string dev_path = devices_dir + name + "/devnum";
            
            std::ifstream bus_file(bus_path);
            std::ifstream dev_file(dev_path);
            
            if (bus_file.is_open() && dev_file.is_open()) {
                int busnum = 0, devnum = 0;
                bus_file >> busnum;
                dev_file >> devnum;
                
                bus_file.close();
                dev_file.close();

                char dev_node_path[128];
                sprintf(dev_node_path, "/dev/bus/usb/%03d/%03d", busnum, devnum);
                std::cout << "[USB Reset] Found Sony device at " << dev_node_path << ". Resetting..." << std::endl;

                int fd = open(dev_node_path, O_WRONLY);
                if (fd >= 0) {
                    if (ioctl(fd, 21780, 0) < 0) {
                        std::cerr << "[USB Reset] IOCTL reset failed: " << strerror(errno) << std::endl;
                    } else {
                        std::cout << "[USB Reset] Reset signal sent successfully!" << std::endl;
                        reset_done = true;
                    }
                    close(fd);
                } else {
                    std::cerr << "[USB Reset] Failed to open " << dev_node_path << ": " << strerror(errno) << std::endl;
                }
            }
        }
    }
    closedir(dir);
    
    if (reset_done) {
        std::cout << "[USB Reset] Waiting 8 seconds for camera to re-initialize..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(8));
        return true;
    }
    return false;
}

std::string format_sdk_code(CrInt32u code) {
    switch(code) {
        case 0: return "CrError_None (0x0000): Success";
        case 0x8200: return "CrError_Connect (0x8200): General connection error";
        case 0x8208: return "CrError_Connect_TimeOut (0x8208): Connection timed out";
        case 0x20000: return "CrWarning_Unknown (0x20000): Unknown warning";
        case 0x20001: return "CrWarning_Connect_Reconnected (0x20001): Connection re-established";
        case 0x20002: return "CrWarning_Connect_Reconnecting (0x20002): Connection lost, attempting to reconnect";
        case 0x20011: return "CrWarning_Connect_Already (0x20011): Connection session already opened / active";
        case 0x20012: return "CrWarning_Connect_OverLimitOfDevice (0x20012): Connection limit exceeded";
        case 0x8400: return "CrError_Api (0x8400): General API call error";
        case 0x8800: return "CrError_Device (0x8800): Device error";
        case 0x8300: return "CrError_Memory (0x8300): Memory or resource error";
        default: {
            char buf[64];
            sprintf(buf, "0x%X", code);
            return std::string(buf);
        }
    }
}

using namespace SCRSDK;

void query_and_set_property(CrDeviceHandle deviceHandle, CrInt32u code, CrInt64u value, const std::string& label) {
    CrInt32u codes[1] = { code };
    CrDeviceProperty* properties = nullptr;
    CrInt32 numOfProperties = 0;
    
    CrError err = GetSelectDeviceProperties(deviceHandle, 1, codes, &properties, &numOfProperties);
    if (err == CrError_None && properties != nullptr && numOfProperties > 0) {
        // Update value using the correct type
        properties[0].SetCurrentValue(value);
        err = SetDeviceProperty(deviceHandle, &properties[0]);
        if (err != CrError_None) {
            std::cerr << "WARNING: Failed to set " << label << ". Code: " << format_sdk_code(err) << std::endl;
        } else {
            std::cout << "Successfully set " << label << std::endl;
        }
        ReleaseDeviceProperties(deviceHandle, properties);
    } else {
        std::cerr << "WARNING: Failed to query " << label << " property from camera. Code: " << format_sdk_code(err) << std::endl;
    }
}

class MyDeviceCallback : public IDeviceCallback {
public:
    MyDeviceCallback() : m_connected(false), m_disconnected(false), m_downloaded(false), m_reconnecting(false), m_downloaded_filename("") {}

    // Called when the camera device connection is successfully established
    virtual void OnConnected(DeviceConnectionVersioin version) override {
        m_connected = true;
        m_reconnecting = false;
        std::cout << "[Callback] Camera connected. Connection version: " << version << std::endl;
    }

    // Called when the camera device is disconnected
    virtual void OnDisconnected(CrInt32u error) override {
        m_disconnected = true;
        std::cout << "[Callback] Camera disconnected. Code: " << format_sdk_code(error) << std::endl;
    }

    // Called when a camera property changes
    virtual void OnPropertyChanged() override {
        std::cout << "[Callback] Property changed on camera." << std::endl;
    }

    // Called when download of a captured file is completed
    virtual void OnCompleteDownload(CrChar* filename, CrInt32u type = 0xFFFFFFFF) override {
        std::string fn(filename);
        std::cout << "[Callback] Download complete: " << fn << " (type: " << type << ")" << std::endl;
        
        // Filter for Sony RAW files (.ARW / .arw)
        if (fn.size() >= 4 && (fn.substr(fn.size() - 4) == ".ARW" || fn.substr(fn.size() - 4) == ".arw")) {
            m_downloaded = true;
            m_downloaded_filename = fn;
        } else {
            // Automatically clean up non-RAW files (like JPEG) if they are transferred
            std::remove(filename);
        }
    }

    // Called for warnings
    virtual void OnWarning(CrInt32u warning) override {
        std::cout << "[Callback] Warning: " << format_sdk_code(warning) << std::endl;
        if (warning == CrWarning_Connect_Reconnecting) {
            m_reconnecting = true;
        }
    }

    // Called for errors
    virtual void OnError(CrInt32u error) override {
        std::cerr << "[Callback] Error occurred: " << format_sdk_code(error) << std::endl;
    }

    bool isConnected() const { return m_connected; }
    bool isDisconnected() const { return m_disconnected; }
    bool isDownloaded() const { return m_downloaded; }
    bool isReconnecting() const { return m_reconnecting; }
    std::string downloadedFilename() const { return m_downloaded_filename; }

    void reset() {
        m_connected = false;
        m_disconnected = false;
        m_downloaded = false;
        m_reconnecting = false;
        m_downloaded_filename = "";
    }

    void resetDownload() {
        m_downloaded = false;
        m_reconnecting = false;
        m_downloaded_filename = "";
    }

private:
    bool m_connected;
    bool m_disconnected;
    bool m_downloaded;
    bool m_reconnecting;
    std::string m_downloaded_filename;
};

// Helper function to run a single capture test case
bool run_capture_test_case(CrDeviceHandle deviceHandle, MyDeviceCallback& callback, 
                           CrInt32u shutterSpeedVal, double expectedShutterSec, 
                           const std::string& label) {
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "TEST CASE: " << label << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 1. Set Shutter Speed property
    std::cout << "Setting Shutter Speed..." << std::endl;
    CrDeviceProperty shutterProp;
    shutterProp.SetCode(CrDeviceProperty_ShutterSpeed);
    shutterProp.SetValueType(CrDataType_UInt32);
    shutterProp.SetCurrentValue(shutterSpeedVal);
    
    CrError err = SetDeviceProperty(deviceHandle, &shutterProp);
    if (err != CrError_None) {
        std::cerr << "ERROR: Failed to set Shutter Speed. Code: " << err << std::endl;
        return false;
    }

    // Wait for setting to apply
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Reset download status in callback
    callback.resetDownload();

    // 2. Trigger Capture
    std::cout << "Triggering camera shutter..." << std::endl;
    err = SendCommand(deviceHandle, CrCommandId_Release, CrCommandParam_Down);
    if (err != CrError_None) {
        std::cerr << "ERROR: Failed to send shutter press down. Code: " << err << std::endl;
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    err = SendCommand(deviceHandle, CrCommandId_Release, CrCommandParam_Up);
    if (err != CrError_None) {
        std::cerr << "ERROR: Failed to send shutter release up. Code: " << err << std::endl;
        return false;
    }

    // 3. Wait for image download
    std::cout << "Waiting for RAW file to download..." << std::endl;
    int waitSeconds = 0;
    while (!callback.isDownloaded() && waitSeconds < 25) {
        if (callback.isDisconnected() || callback.isReconnecting()) {
            std::cout << "[Wait] Connection dropped or reconnecting. Aborting wait." << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        waitSeconds++;
    }

    if (!callback.isDownloaded()) {
        std::cerr << "ERROR: Capture timed out. RAW file not received." << std::endl;
        return false;
    }

    std::string filepath = callback.downloadedFilename();
    std::cout << "Successfully downloaded: " << filepath << std::endl;

    // 4. Load the RAW file and print metadata properties
    std::cout << "Processing RAW file using LibRaw..." << std::endl;
    // Pass debayer = false for speed, since we only need the metadata headers
    LibRaw* proc = RawProcessor::load_raw(filepath, /*debayer*/ false);
    if (proc == nullptr) {
        std::cerr << "ERROR: Failed to load RAW file metadata." << std::endl;
        return false;
    }

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

    // 5. Verify the properties
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

    // Verify ISO (native lowest ISO 100)
    if (std::abs(iso - 100.0) > 1.0) {
        std::cerr << "VERIFICATION FAILURE: Decoded ISO (" << iso 
                  << ") does not match expected base ISO (100)" << std::endl;
        success = false;
    } else {
        std::cout << "  [PASS] ISO matches expected value (100)." << std::endl;
    }

    // 6. Delete the file
    std::cout << "Deleting captured file: " << filepath << std::endl;
    if (std::remove(filepath.c_str()) != 0) {
        std::cerr << "WARNING: Failed to delete " << filepath << std::endl;
    }

    return success;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Sony A7R4 Tethered Capture Test Utility" << std::endl;
    std::cout << "========================================" << std::endl;

    // 0. Auto-reset camera USB if present
    std::cout << "Checking for Sony USB cameras to reset..." << std::endl;
    reset_sony_camera_usb();

    // 1. Initialize the SDK
    std::cout << "Initializing SDK..." << std::endl;
    if (!Init(0)) {
        std::cerr << "ERROR: Failed to initialize Camera Remote SDK." << std::endl;
        return -1;
    }

    // 2. Discover connected cameras
    std::cout << "Scanning for connected cameras..." << std::endl;
    ICrEnumCameraObjectInfo* cameraList = nullptr;
    CrError err = EnumCameraObjects(&cameraList, 3);

    if (err != CrError_None || cameraList == nullptr || cameraList->GetCount() == 0) {
        std::cerr << "ERROR: No cameras found." << std::endl;
        if (cameraList) cameraList->Release();
        Release();
        return -1;
    }

    // Print rich debug details for all found cameras
    std::cout << "\n=== Discovered Cameras Info ===" << std::endl;
    for (CrInt32u i = 0; i < cameraList->GetCount(); i++) {
        const ICrCameraObjectInfo* info = cameraList->GetCameraObjectInfo(i);
        std::cout << "Camera #" << i << ":" << std::endl;
        std::cout << "  Model:               " << (info->GetModel() ? info->GetModel() : "N/A") << std::endl;
        std::cout << "  Name:                " << (info->GetName() ? info->GetName() : "N/A") << std::endl;
        std::cout << "  Connection Type:     " << (info->GetConnectionTypeName() ? info->GetConnectionTypeName() : "N/A") << std::endl;
        std::cout << "  Adaptor Name:        " << (info->GetAdaptorName() ? info->GetAdaptorName() : "N/A") << std::endl;
        std::cout << "  USB PID:             0x" << std::hex << info->GetUsbPid() << std::dec << std::endl;
        std::cout << "  Connection Status:   " << info->GetConnectionStatus() << std::endl;
        std::cout << "  Pairing Necessity:   " << (info->GetPairingNecessity() ? info->GetPairingNecessity() : "N/A") << std::endl;
        std::cout << "  Authentication State:" << info->GetAuthenticationState() << std::endl;
        std::cout << "  SSH Support:         " << info->GetSSHsupport() << std::endl;
        std::cout << "  ID size/type:        " << info->GetIdSize() << " / " << info->GetIdType() << std::endl;
        if (info->GetIPAddressChar()) {
            std::cout << "  IP Address:          " << info->GetIPAddressChar() << std::endl;
        }
        if (info->GetMACAddressChar()) {
            std::cout << "  MAC Address:         " << info->GetMACAddressChar() << std::endl;
        }
    }
    std::cout << "===============================\n" << std::endl;

    // 3. Get the first camera info object
    ICrCameraObjectInfo* cameraInfo = const_cast<ICrCameraObjectInfo*>(cameraList->GetCameraObjectInfo(0));
    if (cameraInfo == nullptr) {
        std::cerr << "ERROR: Failed to retrieve camera information." << std::endl;
        cameraList->Release();
        Release();
        return -1;
    }

    std::cout << "Connecting to camera: " << cameraInfo->GetModel() << "..." << std::endl;

    // 4. Connect to the camera
    MyDeviceCallback callback;
    CrDeviceHandle deviceHandle = 0;

    err = Connect(cameraInfo, &callback, &deviceHandle, CrSdkControlMode_Remote, CrReconnecting_ON);
    if (err != CrError_None) {
        std::cerr << "ERROR: Connect call failed. Code: " << format_sdk_code(err) << std::endl;
        cameraList->Release();
        Release();
        return -1;
    }

    // Wait for OnConnected callback
    std::cout << "Waiting for connection verification..." << std::endl;
    int waitSeconds = 0;
    while (!callback.isConnected() && waitSeconds < 10) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        waitSeconds++;
    }

    if (!callback.isConnected() || deviceHandle == 0) {
        std::cerr << "ERROR: Camera connection timed out or failed. Callback connection state: " 
                  << (callback.isConnected() ? "CONNECTED" : "DISCONNECTED") 
                  << ", Handle: " << deviceHandle << std::endl;
        cameraList->Release();
        Release();
        return -1;
    }

    std::cout << "Connection established successfully!" << std::endl;
    std::cout << "NOTE: Make sure the physical mode dial on the camera is set to M (Manual)." << std::endl;

    // 5. Configure Save Information: save to local directory with prefix "test_capture"
    std::cout << "Configuring save info to local directory..." << std::endl;
    err = SetSaveInfo(deviceHandle, const_cast<char*>("./"), const_cast<char*>("test_capture"), 1);
    if (err != CrError_None) {
        std::cerr << "WARNING: SetSaveInfo failed. Code: " << format_sdk_code(err) << std::endl;
    }

    // Force save destination to Host PC to ensure OnCompleteDownload callback is triggered
    std::cout << "Setting Still Image Store Destination to Host PC..." << std::endl;
    query_and_set_property(deviceHandle, CrDeviceProperty_StillImageStoreDestination, CrStillImageStoreDestination_HostPC, "StillImageStoreDestination");

    // Force Release without Card to Enable to support disk-less scanning
    std::cout << "Setting Release without Card to Enable..." << std::endl;
    query_and_set_property(deviceHandle, CrDeviceProperty_ReleaseWithoutCard, CrReleaseWithoutCard_Enable, "ReleaseWithoutCard");

    // Set camera ISO to 100
    std::cout << "Setting ISO to 100..." << std::endl;
    query_and_set_property(deviceHandle, CrDeviceProperty_IsoSensitivity, 100, "IsoSensitivity");

    bool testSuccess = true;

    // Test Case 1: Shutter Speed = 1.0s
    // 1.0s value: Numerator (upper 16 bits) = 1 (0x0001) | Denominator (lower 16 bits) = 1 (0x0001)
    // 0x00010001 = 65537
    if (!run_capture_test_case(deviceHandle, callback, 0x00010001, 1.0, "1s Exposure")) {
        testSuccess = false;
    }

    // Test Case 2: Shutter Speed = 1/125s
    // 1/125s value: Numerator (upper 16 bits) = 1 (0x0001) | Denominator (lower 16 bits) = 125 (0x007D)
    // 0x0001007D = 65661
    if (!run_capture_test_case(deviceHandle, callback, 0x0001007D, 1.0 / 125.0, "1/125s Exposure")) {
        testSuccess = false;
    }

    // 6. Disconnect and clean up
    std::cout << "\nDisconnecting from camera..." << std::endl;
    if (callback.isDisconnected() || callback.isReconnecting()) {
        std::cout << "[Cleanup] Connection already lost or reconnecting. Skipping Disconnect call to avoid hangs." << std::endl;
    } else {
        Disconnect(deviceHandle);
        waitSeconds = 0;
        while (!callback.isDisconnected() && waitSeconds < 2) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            waitSeconds++;
        }
    }

    ReleaseDevice(deviceHandle);
    cameraList->Release();
    Release();
    std::cout << "SDK shutdown complete." << std::endl;

    if (testSuccess) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "TEST RESULT: SUCCESS" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "TEST RESULT: FAILURE" << std::endl;
        std::cout << "========================================" << std::endl;
        return -1;
    }
}
