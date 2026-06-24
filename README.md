# Negative Film Scanning Station (negicc-station)

This repository contains the software for a negative film scanning station designed to run on the **Nvidia Jetson Nano** (ARMv8 64-bit architecture) and interface with a connected **Sony A7R4** camera.

### Directory Structure

* **[src/](src/)**: Main C++ and CPython extension source code, as well as Python example scripts.
  * [camera_session.h](src/camera_session.h) / [sony_camera_session.cpp](src/sony_camera_session.cpp): Sony Camera Remote SDK wrapper.
  * [raw_processor.cpp](src/raw_processor.cpp): Linear conversion and 4-shot pixel-shift merging.
  * [image_capture.cpp](src/image_capture.cpp): CapturedImage definition and linear TIFF writer.
  * [python_bindings.cpp](src/python_bindings.cpp): CPython bindings exposing tethered capture and raw conversion to Python.
  * [sample_capture_tiff.py](src/sample_capture_tiff.py): Simple command-line capture example.
  * [sample_ui.py](src/sample_ui.py): PyGObject/GTK3 desktop UI application for scanner control and preview.
* **[tests/](tests/)**: Integration tests ([test_cpython.py](tests/test_cpython.py) and [test_live_parity.py](tests/test_live_parity.py)).
* **[test_imgs/](test_imgs/)**: reference RAW images stored using LZMA compression ([test_capture_ref.ARW.xz](test_imgs/test_capture_ref.ARW.xz)).
* **[3rd_party/](3rd_party/)**: Local third-party dependencies, minimal headers, and SDK configuration.
* **[build/](build/)**: Unified folder containing compiled binaries, shared libraries, and build artifacts.
* **[setup.py](setup.py)**: Packaging setup configuration for the C extension.
* **[Makefile](Makefile)**: Target compilation and test harness configurations.

---

## 1. Jetson Nano System Dependencies

Before building and running the scanning software, ensure that the Jetson Nano system is updated and the following system dependencies are installed:

```bash
# Update package list
sudo apt-get update

# Install build tools and C++ compiler
sudo apt-get install -y build-essential g++

# Install SDK and image processing dependencies (USB, XML, LibRaw, and LCMS2),
# PyGObject GUI system dependencies (GObject Introspection, Cairo, and GTK3),
# and Argyll Color Management System (ArgyllCMS) for film profiling
sudo apt-get install -y libusb-1.0-0-dev libxml2-dev libraw-dev liblcms2-dev libgirepository1.0-dev libcairo2-dev gir1.2-gtk-3.0 argyll

```

Additionally, to allow the application to communicate with the Sony camera over USB without requiring superuser (`root`) privileges, configure the USB udev rules as described in the SDK setup guide.

---

## 2. Sony Camera Remote SDK (CrSDK) Integration

The camera remote control capability relies on the proprietary Sony Camera Remote SDK. Because the SDK is proprietary, its headers and libraries are not stored in this repository.

Please follow the detailed setup instructions in **[3rd_party/CrSDK/README.md](3rd_party/CrSDK/README.md)** to download, extract, and install the Linux ARMv8 SDK.

---

## 3. Build and Link Configuration (Makefile Flags)

The project includes a **[Makefile](Makefile)** configured with specific compilation and linking flags optimized for the Jetson Nano (ARM64 architecture) and our library dependencies:

### Compilation Flags (`CXXFLAGS`)
* `-fsigned-char`: **Critical for ARM64 architecture.** By default, `char` is unsigned on ARM64 platforms (unlike x86_64 where it is signed). Since many third-party libraries (including LibRaw headers) expect `char` to be signed, this flag forces `char` to be signed, preventing compilation errors and subtle image parsing bugs.
* `-I3rd_party/CrSDK/include`: Includes the Sony Camera Remote SDK headers.
* `-I3rd_party`: Includes our local third-party headers (such as `lcms2.h` or custom headers).

### Linking Flags (`LDFLAGS`)
* `-L3rd_party/CrSDK/lib -lCr_Core`: Links against the core Sony SDK library.
* `-Wl,-rpath,'$$ORIGIN/3rd_party/CrSDK/lib'`: Sets the run-time shared library search path (rpath) relative to the executable's directory. This allows the application to find `libCr_Core.so` and its adapters at runtime without needing to modify the `LD_LIBRARY_PATH` environment variable.
* `-lraw -llcms2`: Directs the linker to link against `libraw` and `lcms2` system libraries.

> [!IMPORTANT]
> The development packages (`libraw-dev` and `liblcms2-dev`) must be installed on the system beforehand for compilation to succeed. If compilation fails with linker errors like `cannot find -lraw` or `cannot find -llcms2`, make sure you have run:
> ```bash
> sudo apt-get update && sudo apt-get install -y libraw-dev liblcms2-dev
> ```

---

## 4. Building and Running the Project

Once the system dependencies are installed and the Sony SDK files are populated in `3rd_party/CrSDK/`, you can compile the project and run the capture utilities:

```bash
# Build all C++ targets and build/install the Python library
make

# Run the C++ capture test program
./build/capture_test

# Run the command-line Python capture example
./venv/bin/python3 src/sample_capture_tiff.py

# Run the PyGObject GTK3 desktop scanning GUI
./venv/bin/python3 src/sample_ui.py
```

---

## 5. Agent Instructions for Managing Dependencies

When introducing any new third-party dependency, library, or system package to this codebase, the agent **MUST** follow these protocol steps to keep the environment reproducible:

1. **Update System Dependencies**: Add any new system-level package requirements to the **Jetson Nano System Dependencies** section in this top-level [README.md](README.md).
2. **Setup Subdirectory Integration**: If the dependency is a third-party library, create a dedicated folder under `3rd_party/<DependencyName>/` and write a local `README.md` detailing how to download, compile, or install the library.
3. **Configure Git Exclusion**: If the dependency contains proprietary binaries or large compiled libraries, add them to the top-level [.gitignore](.gitignore) to prevent them from being checked into version control.
4. **Document Code & Builds**: Ensure all Makefiles and source files are updated and linked correctly, and document the complete build instructions so another agent can repeat the execution flow on a fresh Jetson Nano environment.

---

## 6. Troubleshooting and Hardware Diagnostics

### A. USB Memory Limit (`ENOMEM` / Disconnection during file transfer)
When transferring large RAW image files (like the 61MP files from the Sony A7R4), the Sony SDK submits multiple concurrent USB requests. If the total size of these requests exceeds the kernel's USB filesystem memory limit, the transfer fails and the connection drops.
* **Symptom**: Callback prints `CrWarning_Connect_Reconnecting (0x20002)` or connection drops, followed by capture timeouts.
* **Diagnostic**: Run the application under `strace` to check for `ENOMEM` errors:
  ```bash
  strace -f -o strace.log ./build/capture_test
  grep "USBDEVFS_SUBMITURB" strace.log
  # Look for: ioctl(..., USBDEVFS_SUBMITURB, ...) = -1 ENOMEM (Cannot allocate memory)
  ```
* **Temporary Fix**: Increase the runtime USB memory limit to 1024MB:
  ```bash
  echo 1024 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
  ```
* **Persistent Fix**: Write the modprobe configuration by running:
  ```bash
  echo "options usbcore usbfs_memory_mb=1024" | sudo tee /etc/modprobe.d/usbcore.conf
  ```

### B. USB Autosuspend
The Linux kernel's USB power management can suspend the camera during long exposures or idle periods, leading to `0x8207` (`CrError_Connect_Disconnected`) errors.
* **Fix**: The application attempts to write `on` to all `/sys/devices/.../power/control` files at startup. Ensure the application is run with sufficient privileges or configure udev rules to disable autosuspend for the camera's USB vendor/product ID.

### C. Stale Sessions (`CrWarning_Connect_Already` / `0x20011`)
Performing a hard USB reset (e.g. `USBDEVFS_RESET` ioctl) right before connecting forces the USB connection to drop without letting the SDK perform its cleanup flow. This leaves a stale session active on the camera side, which causes `CrWarning_Connect_Already (0x20011)` warnings right after capture triggers, aborting the transfer.
* **Fix**: Avoid manual USB resets before connecting; instead, rely on the SDK's built-in reconnection logic (`CrReconnecting_ON`).

### D. Shutter Speed Codes
The Sony SDK represents shutter speed values as a fraction pack where the upper 16 bits are the numerator and the lower 16 bits are the denominator.
* **Fractional Speeds**: Set numerator to `1` and denominator to the fraction value (e.g., `1/125s` is `0x0001007D`).
* **Whole-Second Speeds**: Represented with a **fixed denominator of 10** (`0x000A`) and the numerator scaled accordingly (e.g., `1s` is `10/10` which is `0x000A000A`; `2s` is `20/10` which is `0x0014000A`).
* **Behavior**: If you send a value that is not in the camera's supported list of shutter speeds, the camera will silently reject it and keep its current setting.

---

## 7. Step-by-Step Film Profiling Guide

This guide details the complete protocol to calibrate a digital camera sensor's spectral crosstalk, capture film base characteristics, and profile target data at different exposures using the scanning software.

### Phase 1: Spectral Crosstalk Calibration
Before scanning negative film, you must capture the camera sensor's specific channel overlap characteristics under single-color narrow-band illuminations (or filter bands):

1. Launch the crosstalk calibration script:
   ```bash
   ./venv/bin/python3 src/crosstalk_calibration.py
   ```
2. Place a white diffusion target in the scanning gate.
3. Capture three individual flat-field calibration frames:
   * **Red Frame**: Exposed under narrow-band Red LED illumination.
   * **Green Frame**: Exposed under narrow-band Green LED illumination.
   * **Blue Frame**: Exposed under narrow-band Blue LED illumination.
   * *Note: Run the auto-exposure checker on each to ensure the peak ADU stays below 16384 (no clipping).*
4. Compute the matrix inside the interface and click **Save Crosstalk Profile** to write the calibration details to a JSON file (e.g., `sony_a7r4_crosstalk.json`).

---

### Phase 2: Negative Film Profiling (Multi-Exposure)
With the crosstalk profile ready, you can now scan your film base and IT8 target files to generate a custom self-contained film profile:

1. Launch the main film profiling application:
   ```bash
   ./venv/bin/python3 src/ui_film_profiling.py
   ```
2. Click **LOAD CROSSTALK PROFILE** in the left sidebar and select your `sony_a7r4_crosstalk.json`.
3. In the reference field, input the IT8 target reference URL or local path (e.g. `http://www.colorreference.de/targets/R190808.zip`) and click **DOWNLOAD REFERENCE**.

#### Step A: Capture the Film Base (Orange Mask)
1. Go to the **Film Base** tab.
2. Place a clear, unexposed but developed frame of your negative film (e.g., the orange leader area) in the scanner gate.
3. Use the **Auto-Exposure** button to automatically find the optimal shutter speed (where G/B channels are bright but unclipped).
4. Click **CAPTURE BASE**.
5. Draw a selection box over the orange area in the image preview panel, then click **READ FILM BASE VALUES**.

#### Step B: Capture IT8 Targets at Different Exposures
You can scan multiple targets captured at different exposure offsets (e.g., bracketed scans or exposure sweeps) to map the film's density characteristics accurately:

1. Go to the **Target 1** tab.
2. Place the physical IT8 target film in the scanner gate.
3. Adjust the exposure settings for Target 1 (e.g., shutter speed set to normal $+0$ EV).
4. Click **CAPTURE TARGET**.
5. Align the patch grid:
   * Click **LAYER IT8 MASK** to show the patch mapping boxes.
   * Use the arrow keys (`Up`, `Down`, `Left`, `Right`) to nudge, and `+`/`-` keys to scale the layout until the overlay boxes align precisely with the 288 physical film patches.
6. Click **READ PATCH VALUES**.
7. If you want to profile other exposures (e.g., a $-1$ EV underexposure or a $+1$ EV overexposure):
   * Click the **`+` (Add Tab)** button to create **Target 2**.
   * Adjust the camera shutter speed to the target exposure.
   * Click **CAPTURE TARGET**, align the grid, and click **READ PATCH VALUES**.
   * Repeat this for as many exposure steps as desired.

#### Step C: Generate and Inspect the Profile
1. Click **SAVE PROFILE** in the left sidebar.
2. Enter the name of the film stock (e.g., `Portra400`).
3. Select the file location to save the compiled JSON film profile.
4. A progress dialog modal will appear, displaying active compilation steps:
   * It adapts target values, fits monotonic red/green/blue spline curves to the grayscale patches (the TRC curves), and runs Argyll's `colprof` to generate a custom 3D color lookup table (cLUT) ICC profile.
   * All target coordinates, custom TRC splines, and base64-encoded ICC files are written into a single self-contained JSON profile.
5. Upon completion, the **Report Window** opens automatically, displaying:
   * Vertical tabs for each processed target and a film base summary tab.
   * Subplots showing the generated TRC curves and characteristic D-log H curves.
   * The final profile verification error metrics (max/average CIEDE2000 errors).
   * A static **Target Converted** positive image showing the crosstalk-corrected and color-managed positive result.
   * Collapsible compilation step logs.
