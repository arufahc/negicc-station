#include <iostream>
#include <chrono>
#include <thread>
#include "CameraRemote_SDK.h"
#include "IDeviceCallback.h"
#include "ICrCameraObjectInfo.h"

using namespace SCRSDK;

class MyDeviceCallback : public IDeviceCallback {
public:
    MyDeviceCallback() : m_connected(false), m_disconnected(false) {}

    // Called when the camera device connection is successfully established
    virtual void OnConnected(DeviceConnectionVersioin version) override {
        m_connected = true;
        std::cout << "[Callback] Camera connected. Connection version: " << version << std::endl;
    }

    // Called when the camera device is disconnected
    virtual void OnDisconnected(CrInt32u error) override {
        m_disconnected = true;
        std::cout << "[Callback] Camera disconnected. Error code: " << error << std::endl;
    }

    // Called when a camera property changes
    virtual void OnPropertyChanged() override {
        std::cout << "[Callback] Property changed on camera." << std::endl;
    }

    // Called for warnings
    virtual void OnWarning(CrInt32u warning) override {
        std::cout << "[Callback] Warning: 0x" << std::hex << warning << std::dec << std::endl;
    }

    // Called for errors
    virtual void OnError(CrInt32u error) override {
        std::cerr << "[Callback] Error occurred: 0x" << std::hex << error << std::dec << std::endl;
    }

    bool isConnected() const { return m_connected; }
    bool isDisconnected() const { return m_disconnected; }

    void reset() {
        m_connected = false;
        m_disconnected = false;
    }

private:
    bool m_connected;
    bool m_disconnected;
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Sony Camera Remote SDK Capture Tool" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Initialize the SDK
    std::cout << "Initializing SDK..." << std::endl;
    if (!Init(0)) {
        std::cerr << "ERROR: Failed to initialize Camera Remote SDK." << std::endl;
        return -1;
    }

    // 2. Discover connected cameras
    std::cout << "Scanning for connected cameras..." << std::endl;
    ICrEnumCameraObjectInfo* cameraList = nullptr;
    CrError err = EnumCameraObjects(&cameraList, 3); // Wait up to 3 seconds

    if (err != CrError_None || cameraList == nullptr || cameraList->GetCount() == 0) {
        std::cerr << "ERROR: No cameras found. Please verify the physical USB connection and that the camera's mode is set to 'PC Remote'." << std::endl;
        if (cameraList) cameraList->Release();
        Release();
        return -1;
    }

    int count = cameraList->GetCount();
    std::cout << "Found " << count << " camera(s)." << std::endl;

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
        std::cerr << "ERROR: Connect call failed. Code: " << err << std::endl;
        cameraList->Release();
        Release();
        return -1;
    }

    // Wait for OnConnected callback to verify connection is established
    std::cout << "Waiting for connection verification..." << std::endl;
    int waitSeconds = 0;
    while (!callback.isConnected() && waitSeconds < 10) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        waitSeconds++;
    }

    if (!callback.isConnected() || deviceHandle == 0) {
        std::cerr << "ERROR: Camera connection timed out or handle is invalid." << std::endl;
        cameraList->Release();
        Release();
        return -1;
    }

    std::cout << "Connection established successfully!" << std::endl;
    std::cout << "NOTE: Make sure the physical mode dial on the camera is set to M (Manual)." << std::endl;

    // 5. Configure settings: Shutter Speed = 1/125s
    std::cout << "Setting Shutter Speed to 1/125s..." << std::endl;
    CrDeviceProperty shutterProp;
    shutterProp.SetCode(CrDeviceProperty_ShutterSpeed);
    shutterProp.SetValueType(CrDataType_UInt32);
    // Value is 32-bit: Numerator (upper 16 bits = 0x0001) | Denominator (lower 16 bits = 125 = 0x007D)
    // 0x0001007D = 65661
    shutterProp.SetCurrentValue(0x0001007D); 
    
    err = SetDeviceProperty(deviceHandle, &shutterProp);
    if (err != CrError_None) {
        std::cerr << "WARNING: Failed to apply Shutter Speed change. Code: " << err << std::endl;
    }

    // 6. Configure settings: ISO = 100 (lowest native ISO)
    std::cout << "Setting ISO to 100..." << std::endl;
    CrDeviceProperty isoProp;
    isoProp.SetCode(CrDeviceProperty_IsoSensitivity);
    isoProp.SetValueType(CrDataType_UInt32);
    // Bits 0-23 represent the ISO value directly
    isoProp.SetCurrentValue(100); 

    err = SetDeviceProperty(deviceHandle, &isoProp);
    if (err != CrError_None) {
        std::cerr << "WARNING: Failed to apply ISO change. Code: " << err << std::endl;
    }

    // Wait 2 seconds for settings to propagate to the device
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 7. Take a picture
    std::cout << "Triggering camera shutter..." << std::endl;
    
    // Simulate Shutter Press Down
    err = SendCommand(deviceHandle, CrCommandId_Release, CrCommandParam_Down);
    if (err != CrError_None) {
        std::cerr << "ERROR: Failed to send shutter press down. Code: " << err << std::endl;
    }

    // Hold the shutter button down for 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Simulate Shutter Release Up
    err = SendCommand(deviceHandle, CrCommandId_Release, CrCommandParam_Up);
    if (err != CrError_None) {
        std::cerr << "ERROR: Failed to send shutter release up. Code: " << err << std::endl;
    } else {
        std::cout << "Photo captured successfully!" << std::endl;
    }

    // Wait 3 seconds for photo storage/processing to complete on the camera
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 8. Disconnect and clean up
    std::cout << "Disconnecting from camera..." << std::endl;
    Disconnect(deviceHandle);

    // Wait for OnDisconnected callback
    waitSeconds = 0;
    while (!callback.isDisconnected() && waitSeconds < 5) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        waitSeconds++;
    }

    // Release connection resources
    ReleaseDevice(deviceHandle);
    std::cout << "Camera connection released." << std::endl;

    // Release camera list and SDK resources
    cameraList->Release();
    Release();
    std::cout << "SDK shutdown complete." << std::endl;

    return 0;
}
