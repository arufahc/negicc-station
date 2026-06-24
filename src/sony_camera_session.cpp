#include "sony_camera_session.h"
#include "CameraRemote_SDK.h"
#include "IDeviceCallback.h"
#include "ICrCameraObjectInfo.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <cstdlib>

using namespace SCRSDK;

// Helper function to format SDK codes (internal to this translation unit)
static std::string format_sdk_code(CrInt32u code) {
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

// Device callback class implementing Sony SDK callbacks
class SonyDeviceCallback : public IDeviceCallback {
public:
    SonyDeviceCallback() : m_connected(false), m_disconnected(false), m_downloaded(false), m_reconnecting(false), m_downloaded_filename("") {}

    virtual void OnConnected(DeviceConnectionVersioin version) override {
        m_connected = true;
        m_reconnecting = false;
        std::cout << "[Callback] Camera connected. Connection version: " << version << std::endl;
    }

    virtual void OnDisconnected(CrInt32u error) override {
        m_disconnected = true;
        std::cout << "[Callback] Camera disconnected. Code: " << format_sdk_code(error) << std::endl;
    }

    virtual void OnPropertyChanged() override {
        std::cout << "[Callback] Property changed on camera." << std::endl;
    }

    virtual void OnCompleteDownload(CrChar* filename, CrInt32u type = 0xFFFFFFFF) override {
        std::string fn(filename);
        std::cout << "[Callback] Download complete: " << fn << " (type: " << type << ")" << std::endl;
        
        if (fn.size() >= 4 && (fn.substr(fn.size() - 4) == ".ARW" || fn.substr(fn.size() - 4) == ".arw")) {
            m_downloaded = true;
            m_downloaded_filename = fn;
        } else {
            std::remove(filename);
        }
    }

    virtual void OnNotifyContentsTransfer(CrInt32u notify, CrContentHandle handle, CrChar* filename = 0) override {
        std::cout << "[Callback] OnNotifyContentsTransfer: notify=" << format_sdk_code(notify)
                  << ", handle=" << handle
                  << ", filename=" << (filename ? filename : "nullptr") << std::endl;
    }

    virtual void OnNotifyRemoteTransferResult(CrInt32u notify, CrInt32u per, CrChar* filename) override {
        std::cout << "[Callback] OnNotifyRemoteTransferResult (file): notify=" << format_sdk_code(notify)
                  << ", per=" << per << "%, filename=" << (filename ? filename : "nullptr") << std::endl;
        if (filename && notify == 1) {
            std::string fn(filename);
            if (fn.size() >= 4 && (fn.substr(fn.size() - 4) == ".ARW" || fn.substr(fn.size() - 4) == ".arw")) {
                m_downloaded = true;
                m_downloaded_filename = fn;
            }
        }
    }

    virtual void OnNotifyRemoteTransferResult(CrInt32u notify, CrInt32u per, CrInt8u* data, CrInt64u size) override {
        std::cout << "[Callback] OnNotifyRemoteTransferResult (data): notify=" << format_sdk_code(notify)
                  << ", per=" << per << "%, data=" << (void*)data << ", size=" << size << " bytes" << std::endl;
    }

    virtual void OnWarning(CrInt32u warning) override {
        std::cout << "[Callback] Warning: " << format_sdk_code(warning) << std::endl;
        if (warning == CrWarning_Connect_Reconnecting) {
            m_reconnecting = true;
        }
    }

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

// Impl structure containing private Sony variables
struct SonyCameraSession::Impl {
    CrDeviceHandle deviceHandle = 0;
    SonyDeviceCallback callback;
    ICrEnumCameraObjectInfo* cameraList = nullptr;
    ICrCameraObjectInfo* cameraInfo = nullptr;
    bool sdkInitialized = false;
    uint32_t currentShutterSpeed = 0x0001007D; // Default to 1/125s
};

SonyCameraSession::SonyCameraSession() : m_impl(std::make_unique<Impl>()) {}

SonyCameraSession::~SonyCameraSession() {
    close();
}

static void disable_usb_autosuspend() {
    std::cout << "[USB] Disabling USB autosuspend..." << std::endl;
    int ret = system("find /sys/devices/ -name 'control' -path '*/power/control' "
                     "-exec sh -c 'echo on > \"{}\"' ';' 2>/dev/null");
    if (ret != 0) {
        std::cerr << "[USB] Warning: autosuspend disable returned non-zero (may need root)." << std::endl;
    } else {
        std::cout << "[USB] Autosuspend disabled." << std::endl;
    }
}

bool SonyCameraSession::initialize() {
    disable_usb_autosuspend();

    std::cout << "Initializing SDK..." << std::endl;
    if (!Init(0)) {
        std::cerr << "ERROR: Failed to initialize Camera Remote SDK." << std::endl;
        return false;
    }
    m_impl->sdkInitialized = true;

    std::cout << "Scanning for connected cameras..." << std::endl;
    CrError err = EnumCameraObjects(&m_impl->cameraList, 3);
    if (err != CrError_None || m_impl->cameraList == nullptr || m_impl->cameraList->GetCount() == 0) {
        std::cerr << "ERROR: No cameras found." << std::endl;
        return false;
    }

    std::cout << "\n=== Discovered Cameras Info ===" << std::endl;
    for (CrInt32u i = 0; i < m_impl->cameraList->GetCount(); i++) {
        const ICrCameraObjectInfo* info = m_impl->cameraList->GetCameraObjectInfo(i);
        std::cout << "Camera #" << i << ":" << std::endl;
        std::cout << "  Model:               " << (info->GetModel() ? info->GetModel() : "N/A") << std::endl;
        std::cout << "  Connection Type:     " << (info->GetConnectionTypeName() ? info->GetConnectionTypeName() : "N/A") << std::endl;
        std::cout << "  Adaptor Name:        " << (info->GetAdaptorName() ? info->GetAdaptorName() : "N/A") << std::endl;
        std::cout << "  USB PID:             0x" << std::hex << info->GetUsbPid() << std::dec << std::endl;
        std::cout << "  Connection Status:   " << info->GetConnectionStatus() << std::endl;
    }
    std::cout << "===============================\n" << std::endl;

    m_impl->cameraInfo = const_cast<ICrCameraObjectInfo*>(m_impl->cameraList->GetCameraObjectInfo(0));
    if (m_impl->cameraInfo == nullptr) {
        std::cerr << "ERROR: Failed to retrieve camera information." << std::endl;
        return false;
    }

    std::cout << "Connecting to camera: " << m_impl->cameraInfo->GetModel() << "..." << std::endl;
    err = Connect(m_impl->cameraInfo, &m_impl->callback, &m_impl->deviceHandle, CrSdkControlMode_Remote, CrReconnecting_ON);
    if (err != CrError_None) {
        std::cerr << "ERROR: Connect call failed. Code: " << format_sdk_code(err) << std::endl;
        return false;
    }

    std::cout << "Waiting for connection verification..." << std::endl;
    int waitSeconds = 0;
    while (!m_impl->callback.isConnected() && waitSeconds < 10) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        waitSeconds++;
    }

    if (!m_impl->callback.isConnected() || m_impl->deviceHandle == 0) {
        std::cerr << "ERROR: Camera connection timed out or failed." << std::endl;
        return false;
    }

    std::cout << "Connection established successfully!" << std::endl;
    return true;
}

static void query_and_set_property(CrDeviceHandle deviceHandle, CrInt32u code, CrInt64u value, const std::string& label) {
    CrInt32u codes[1] = { code };
    CrDeviceProperty* properties = nullptr;
    CrInt32 numOfProperties = 0;
    
    CrError err = GetSelectDeviceProperties(deviceHandle, 1, codes, &properties, &numOfProperties);
    if (err == CrError_None && properties != nullptr && numOfProperties > 0) {
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

static void query_and_print_battery_info(CrDeviceHandle deviceHandle) {
    CrInt32u codes[2] = { CrDeviceProperty_BatteryRemain, CrDeviceProperty_BatteryLevel };
    CrDeviceProperty* properties = nullptr;
    CrInt32 thereAreProperties = 0;
    
    CrError err = GetSelectDeviceProperties(deviceHandle, 2, codes, &properties, &thereAreProperties);
    if (err == CrError_None && properties != nullptr && thereAreProperties > 0) {
        for (int i = 0; i < thereAreProperties; i++) {
            if (properties[i].GetCode() == CrDeviceProperty_BatteryRemain) {
                std::cout << "[Battery] Remaining capacity: " << properties[i].GetCurrentValue() << "%" << std::endl;
            } else if (properties[i].GetCode() == CrDeviceProperty_BatteryLevel) {
                std::cout << "[Battery] Level indicator code: 0x" << std::hex << properties[i].GetCurrentValue() << std::dec << std::endl;
            }
        }
        ReleaseDeviceProperties(deviceHandle, properties);
    } else {
        std::cerr << "WARNING: Failed to query Battery info. Code: " << format_sdk_code(err) << std::endl;
    }
}

static void query_and_print_supported_shutter_speeds(CrDeviceHandle deviceHandle) {
    CrInt32u codes[1] = { CrDeviceProperty_ShutterSpeed };
    CrDeviceProperty* properties = nullptr;
    CrInt32 numOfProperties = 0;
    
    CrError err = GetSelectDeviceProperties(deviceHandle, 1, codes, &properties, &numOfProperties);
    if (err == CrError_None && properties != nullptr && numOfProperties > 0) {
        CrDeviceProperty& prop = properties[0];
        std::cout << "[ShutterSpeed] Current value: 0x" << std::hex << prop.GetCurrentValue() << std::dec << std::endl;
        
        CrDataType type = prop.GetValueType();
        std::cout << "[ShutterSpeed] ValueType: 0x" << std::hex << type << std::dec << std::endl;
        
        CrInt32u valSize = prop.GetValueSize();
        CrInt8u* valPtr = prop.GetValues();
        
        CrInt32u setSize = prop.GetSetValueSize();
        CrInt8u* setPtr = prop.GetSetValues();
        
        CrInt8u* activePtr = (setPtr != nullptr) ? setPtr : valPtr;
        CrInt32u activeSize = (setPtr != nullptr) ? setSize : valSize;
        CrInt32u elementCount = activeSize / sizeof(CrInt32u);
        
        if (activePtr != nullptr && elementCount > 0) {
            CrInt32u* vals = (CrInt32u*)activePtr;
            for (CrInt32u i = 0; i < elementCount; i++) {
                CrInt32u val = vals[i];
                CrInt16u numerator = val >> 16;
                CrInt16u denominator = val & 0xFFFF;
                std::cout << "  - 0x" << std::hex << val << std::dec 
                          << " (Hex) -> " << numerator << "/" << denominator << "s";
                if (denominator == 10) {
                    std::cout << " (decimal " << (double)numerator / 10.0 << "s)";
                } else if (denominator > 0) {
                    std::cout << " (fraction " << (double)numerator / (double)denominator << "s)";
                }
                std::cout << std::endl;
            }
        }
        ReleaseDeviceProperties(deviceHandle, properties);
    }
}

bool SonyCameraSession::configure_settings() {
    if (m_impl->deviceHandle == 0) return false;

    std::string abs_path = std::filesystem::temp_directory_path().string();
    if (abs_path.empty() || abs_path.back() != '/') {
        abs_path += "/";
    }
    std::cout << "Configuring save info to absolute directory: " << abs_path << std::endl;
    CrError err = SetSaveInfo(m_impl->deviceHandle, const_cast<char*>(abs_path.c_str()), const_cast<char*>("test_capture"), 1);
    if (err != CrError_None) {
        std::cerr << "WARNING: SetSaveInfo failed. Code: " << format_sdk_code(err) << std::endl;
    }

    std::cout << "Querying battery status..." << std::endl;
    query_and_print_battery_info(m_impl->deviceHandle);

    std::cout << "Querying supported shutter speeds..." << std::endl;
    query_and_print_supported_shutter_speeds(m_impl->deviceHandle);

    std::cout << "Setting Still Image Store Destination to Host PC..." << std::endl;
    query_and_set_property(m_impl->deviceHandle, CrDeviceProperty_StillImageStoreDestination, CrStillImageStoreDestination_HostPC, "StillImageStoreDestination");

    std::cout << "Setting ISO to 100..." << std::endl;
    query_and_set_property(m_impl->deviceHandle, CrDeviceProperty_IsoSensitivity, 100, "IsoSensitivity");

    return true;
}

bool SonyCameraSession::set_shutter_speed(uint32_t val) {
    if (m_impl->deviceHandle == 0) return false;
    std::cout << "Setting Shutter Speed to 0x" << std::hex << val << std::dec << std::endl;
    CrDeviceProperty shutterProp;
    shutterProp.SetCode(CrDeviceProperty_ShutterSpeed);
    shutterProp.SetValueType(CrDataType_UInt32);
    shutterProp.SetCurrentValue(val);
    
    CrError err = SetDeviceProperty(m_impl->deviceHandle, &shutterProp);
    if (err != CrError_None) {
        std::cerr << "ERROR: Failed to set Shutter Speed. Code: " << err << std::endl;
        return false;
    }
    m_impl->currentShutterSpeed = val;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return true;
}

bool SonyCameraSession::capture(CaptureType type, CaptureOutput& output) {
    if (m_impl->deviceHandle == 0) return false;
    output.type = type;
    output.filepaths.clear();

    // Calculate dynamic shutter press duration based on set shutter speed
    uint32_t shutterVal = m_impl->currentShutterSpeed;
    uint16_t numerator = shutterVal >> 16;
    uint16_t denominator = shutterVal & 0xFFFF;
    double shutterSec = (denominator > 0) ? (double)numerator / (double)denominator : 0.1;
    int exposureMs = (int)(shutterSec * 1000.0);
    int sleepMs = std::max(100, exposureMs + 100);

    int numShots = (type == CaptureType::SONY_PIXEL_SHIFT_4) ? 4 : 1;
    
    for (int shot = 0; shot < numShots; ++shot) {
        if (numShots > 1) {
            std::cout << "\n--- Pixel Shift Shot " << (shot + 1) << " of " << numShots << " ---" << std::endl;
        }
        
        m_impl->callback.resetDownload();

        std::cout << "[Capture] Triggering camera shutter...DOWN" << std::endl;
        CrError err = SendCommand(m_impl->deviceHandle, CrCommandId_Release, CrCommandParam_Down);
        if (err != CrError_None) {
            std::cerr << "ERROR: Failed to send shutter press down. Code: " << err << std::endl;
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));

        std::cout << "[Capture] Triggering camera shutter...UP" << std::endl;
        err = SendCommand(m_impl->deviceHandle, CrCommandId_Release, CrCommandParam_Up);
        if (err != CrError_None) {
            std::cerr << "ERROR: Failed to send shutter release up. Code: " << err << std::endl;
            return false;
        }

        std::cout << "Waiting for RAW file to download..." << std::endl;
        int waitSeconds = 0;
        while (!m_impl->callback.isDownloaded() && waitSeconds < 25) {
            if (m_impl->callback.isDisconnected()) {
                std::cout << "[Wait] Connection lost (0x8207). Aborting wait." << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            waitSeconds++;
        }

        if (!m_impl->callback.isDownloaded()) {
            std::cerr << "ERROR: Capture timed out. RAW file not received." << std::endl;
            return false;
        }

        std::string filepath = m_impl->callback.downloadedFilename();
        std::cout << "Successfully downloaded: " << filepath << std::endl;
        output.filepaths.push_back(filepath);

        // Sleep between shots if doing multi-shot/pixel shift to allow sensor settle / processing
        if (numShots > 1 && shot < numShots - 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    return true;
}

void SonyCameraSession::close() {
    if (m_impl->deviceHandle != 0) {
        std::cout << "\nDisconnecting from camera..." << std::endl;
        if (m_impl->callback.isDisconnected() || m_impl->callback.isReconnecting()) {
            std::cout << "[Cleanup] Connection lost or reconnecting. Exiting immediately to avoid hangs." << std::endl;
        } else {
            Disconnect(m_impl->deviceHandle);
            int waitSeconds = 0;
            while (!m_impl->callback.isDisconnected() && waitSeconds < 2) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                waitSeconds++;
            }
            ReleaseDevice(m_impl->deviceHandle);
            std::cout << "SDK device released." << std::endl;
        }
        m_impl->deviceHandle = 0;
    }

    if (m_impl->cameraList != nullptr) {
        m_impl->cameraList->Release();
        m_impl->cameraList = nullptr;
    }

    if (m_impl->sdkInitialized) {
        Release();
        m_impl->sdkInitialized = false;
        std::cout << "SDK shutdown complete." << std::endl;
    }
}
