# Sony Camera Remote SDK (CrSDK) Integration

This directory ([3rd_party/CrSDK](file:///home/alpha/Projects/negicc-station/3rd_party/CrSDK)) is designated to house the Sony Camera Remote SDK (CrSDK) binaries and headers. Since the SDK is proprietary and subject to Sony's licensing terms, it cannot be version-controlled directly in this repository.

This project is a negative film scanning software designed to interface with Sony cameras (specifically the **Sony A7R4** / ILCE-7RM4) and is built to run on the **Nvidia Jetson Nano** (ARMv8 64-bit architecture).

---

## 1. Download the SDK

1. Go to the [Sony Camera Remote SDK License Agreement Page](https://support.d-imaging.sony.co.jp/app/sdk/licenseagreement_d/en.html).
2. Review and accept the terms of the agreement.
3. Fill out the registration form if prompted.
4. Select and download the **Linux 64bit (ARMv8)** version of the SDK. The downloaded file will typically be a ZIP archive named `CrSDK_vX.XX.XX_YYYYMMDDx.zip`.

---

## 2. Extract and Populate the Directory

Extract the contents of the downloaded ZIP archive and copy the required files into this directory (`3rd_party/CrSDK`) to match the structure below.

### Expected Directory Layout

Once populated, the directory must have the following structure:

```
3rd_party/CrSDK/
├── README.md (This file)
├── include/
│   ├── CrCameraCommonDefine.h
│   ├── CrCommandParam.h
│   ├── CrDefine.h
│   ├── CrError.h
│   ├── CrInterface.h
│   └── ICrCameraObjectInfo.h
└── lib/
    ├── libCr_Core.so
    ├── libmonitor_protocol.so
    ├── libmonitor_protocol_pf.so
    └── CrAdapter/
        ├── libCr_PTP_IP.so
        └── libCr_PTP_USB.so
```

### File Extraction Mapping

Copy the files from the extracted SDK folder as follows:

1. **Headers**:
   Copy all files from the SDK's `include` directory to:
   * [include/](file:///home/alpha/Projects/negicc-station/3rd_party/CrSDK/include)

2. **Libraries**:
   Create a new directory named `lib` at `3rd_party/CrSDK/lib/` and copy the contents of the SDK's `app` (or `libs/linux/armv8`) folder there:
   * Copy `libCr_Core.so`, `libmonitor_protocol.so`, and `libmonitor_protocol_pf.so` to [lib/](file:///home/alpha/Projects/negicc-station/3rd_party/CrSDK/lib).
   * Copy the entire `CrAdapter` directory (containing `libCr_PTP_IP.so` and `libCr_PTP_USB.so`) to [lib/CrAdapter/](file:///home/alpha/Projects/negicc-station/3rd_party/CrSDK/lib/CrAdapter).

---

## 3. Jetson Nano Configuration

### USB Access Permissions (udev rule)
By default, standard Linux users do not have permissions to access raw USB devices like the Sony camera. To avoid running the film scanning software as `root`, configure a `udev` rule:

1. Create a new rules file:
   ```bash
   sudo nano /etc/udev/rules.d/99-sony-camera.rules
   ```
2. Add the following rule (Sony Vendor ID is `054c`):
   ```udev
   SUBSYSTEM=="usb", ATTR{idVendor}=="054c", MODE="0666", GROUP="plugdev"
   ```
3. Reload the `udev` rules:
   ```bash
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
4. Ensure your user belongs to the `plugdev` group:
   ```bash
   sudo usermod -aG plugdev $USER
   ```
   *(Note: You may need to log out and log back in for group changes to take effect).*

### Runtime Library Discovery
The application needs to know where to find `libCr_Core.so` and its adapters at runtime.
- **Development**: Ensure your build system (e.g. CMake) sets the appropriate `RPATH` pointing to the library folder, or set the `LD_LIBRARY_PATH` before running the executable:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alpha/Projects/negicc-station/3rd_party/CrSDK/lib
  ```

### System Dependencies
Ensure the following packages are installed on the Jetson Nano:
```bash
sudo apt-get update
sudo apt-get install -y libusb-1.0-0-dev libxml2-dev
```
