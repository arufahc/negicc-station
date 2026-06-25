# Negative Film Scanning Station (negicc-station)

This repository contains the software for a negative film scanning station designed to run on the **Nvidia Jetson Nano** (ARMv8 64-bit architecture) and interface with a connected **Sony A7R4** camera.

### Directory Structure

* **[src/](src/)**: Main C++ and CPython extension source code, as well as Python example scripts.
  * [camera_session.h](src/camera_session.h) / [sony_camera_session.cpp](src/sony_camera_session.cpp): Sony Camera Remote SDK wrapper.
  * [raw_processor.cpp](src/raw_processor.cpp): Linear conversion and 4-shot pixel-shift merging.
  * [image_capture.cpp](src/image_capture.cpp): CapturedImage definition and linear TIFF writer.
  * [python_bindings.cpp](src/python_bindings.cpp): CPython bindings exposing tethered capture and raw conversion to Python.
  * [sample_capture_tiff.py](src/sample_capture_tiff.py): Simple command-line capture example.
  * [ui_capture.py](src/ui_capture.py): PyGObject/GTK3 desktop UI application for scanner control and preview.
* **[tests/](tests/)**: Integration tests ([test_cpython.py](tests/test_cpython.py) and [test_live_parity.py](tests/test_live_parity.py)).
* **[test_imgs/](test_imgs/)**: reference RAW images stored using LZMA compression ([test_capture_ref.ARW.xz](test_imgs/test_capture_ref.ARW.xz)).
* **[3rd_party/](3rd_party/)**: Local third-party dependencies, minimal headers, and SDK configuration.
* **[build/](build/)**: Unified folder containing compiled binaries, shared libraries, and build artifacts.
* **[setup.py](setup.py)**: Packaging setup configuration for the C extension.
* **[Makefile](Makefile)**: Target compilation and test harness configurations.

---

## Why This Setup is Superior (Scientific Principles)

This scanning station employs a mathematically rigorous pipeline to digitize negative film, yielding superior results compared to traditional flatbed scanners or simple camera-on-copy-stand captures. 

### 1. Light Source & Sensor Crosstalk Calibration
Even when utilizing high-quality narrow-band LED light sources, the camera sensor's built-in color filter array (CFA) has relatively wide, overlapping spectral sensitivity curves (crosstalk). As a result, red light registers slightly on the green pixels, green light registers on red and blue, and so forth.
* **The Solution**: This system calibrates the camera sensor's color matrix with respect to the specific light source. By capturing flat-field red, green, and blue exposures, we construct and apply a crosstalk correction matrix that mathematically decouples the channel overlaps, ensuring true channel purity before any color profiling or negative inversion.

### 2. Film Stock Profiling & Color Management
Correcting for crosstalk alone is insufficient. Without profiling for the specific film stock (e.g., Kodak Portra 400, Fujifilm Gold 200), the raw RGB values—even if corrected—lack color space metadata (like ICC profiles) and will not represent true-to-life colors.
* **The Solution**: The profiling module fits monotonic red, green, and blue spline curves (TRC curves) to IT8 target grayscale patches and compiles a custom 3D color lookup table (cLUT) ICC profile. This maps the sensor's raw response to a standard colorimetric space, preserving color fidelity and rendering the film stock's signature colors accurately.

### 3. Film Base Normalization (Transmittance Scaling)
Negative film possesses an orange-tinted developed emulsion (the film base). Capturing and normalizing the film base is critical to isolate the actual image information.
* **The Solution**: The system measures the film base to normalize all scanned transmittances. Since the film base represents the maximum possible light transmission (unexposed film), treating it as the $100\%$ transmittance reference ($1.0$ in linear space) allows us to compute all subsequent exposure transmittances relative to the base, effectively neutralizing the orange mask without discarding dynamic range.

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

### USB udev Rules (Passwordless Camera Access)

By default, USB device nodes are owned by `root`. Without a `udev` rule the application will receive `Permission denied` errors when it tries to open the camera. Follow these steps to grant your user account direct USB access:

**1. Find the Sony camera's USB Vendor ID and Product ID**

Plug the Sony A7R4 in via USB and run:
```bash
lsusb | grep -i sony
```
Example output:
```
Bus 001 Device 003: ID 054c:0cf3 Sony Corp. ILCE-7RM4
```
In this example the **Vendor ID (VID)** is `054c` and the **Product ID (PID)** is `0cf3`.
The VID for all Sony cameras is always `054c`; the PID may differ by model.

**2. Create the udev rule file**

```bash
sudo nano /etc/udev/rules.d/90-sony-camera.rules
```

Add the following line, replacing `054c` and `0cf3` with the VID and PID from step 1:
```
SUBSYSTEM=="usb", ATTR{idVendor}=="054c", ATTR{idProduct}=="0cf3", MODE="0660", GROUP="plugdev"
```

This sets the device node to be readable/writable (`0660`) by members of the `plugdev` group.

**3. Add your user to the `plugdev` group**

```bash
sudo usermod -aG plugdev $USER
```

> [!IMPORTANT]
> You must **log out and log back in** (or reboot) for the new group membership to take effect.

**4. Reload udev rules and re-trigger**

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**5. Verify access**

Unplug and replug the camera, then confirm the device node is accessible without `sudo`:
```bash
# List connected Sony USB devices — should appear without error
lsusb | grep -i sony

# Check the device node permissions (replace XXX/YYY with bus/device numbers from lsusb)
ls -la /dev/bus/usb/XXX/YYY
# Expected:  crw-rw---- 1 root plugdev ... /dev/bus/usb/XXX/YYY
```

If the group shows `plugdev` and your user is in that group, the application can communicate with the camera over USB without `sudo`.

---

## 2. Sony Camera Remote SDK (CrSDK) Integration

The camera remote control capability relies on the proprietary Sony Camera Remote SDK. Because the SDK is proprietary, its headers and libraries are not stored in this repository.

Please follow the detailed setup instructions in **[3rd_party/CrSDK/README.md](3rd_party/CrSDK/README.md)** to download, extract, and install the Linux ARMv8 SDK.

---

## 3. Build Targets

Once the system dependencies are installed and the Sony SDK files are populated in `3rd_party/CrSDK/`, build and run the project using the [Makefile](Makefile):

| Target | Description |
|---|---|
| `make` | *(default)* Builds all C++ binaries and installs the Python extension library into `venv/`. |
| `make python_lib` | Builds and installs only the Python C extension (skips C++ standalone binaries). |
| `make test_parity` | Runs the pixel-level parity integration test between C++ and Python outputs. |
| `make test_live` | Runs the live camera capture integration test (requires camera connected). |
| `make profile_gen_dry_run` | Dry-runs the profile build pipeline against an existing `.json` profile and IT8 reference — prints metrics without saving. Edit the profile path in the Makefile before use. |
| `make profile_gen_dry_run_graph` | Same as above but also opens debug plots of the TRC curves and D-log H curves. |
| `make profile_gen_and_convert` | Converts `sample.ARW` using an existing profile, writing the result to `build/sample_converted.tiff`. Decompresses `test_imgs/sample_portra400.ARW.xz` automatically if `sample.ARW` is absent. Edit the profile path in the Makefile before use. |
| `make compare_pipelines` | Runs a multi-backend comparison benchmark on a sample RAW image, contrasting Python, C++ CPU, and CUDA execution speeds and logging parity metrics. |
| `make clean` | Removes all build artifacts from `build/`. |

```bash
# Build everything
make

# Run the central UI launcher (recommended)
./venv/bin/python3 src/ui_main.py
```

### Desktop Launcher Installation

To make it easy to start the scanning application from your desktop, you can install a `.desktop` launcher. This launcher runs the interface inside the virtual environment (`venv`) and appends all launch times, standard output, and standard error logs to `~/.config/negicc-station/log`.

To generate and install the desktop shortcut, run the following commands from the repository root:

```bash
# Create the launcher shortcut on your Desktop
cat <<EOF > ~/Desktop/negicc-station.desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=NegICC Station
Comment=Launch NegICC Capture Station
Exec=bash -c "mkdir -p \$HOME/.config/negicc-station && echo '--- Launch: \$(date) ---' >> \$HOME/.config/negicc-station/log && $(pwd)/venv/bin/python $(pwd)/src/ui_main.py >> \$HOME/.config/negicc-station/log 2>&1"
Icon=$(pwd)/src/camera_icon.jpg
Path=$(pwd)
Terminal=false
Categories=Graphics;Utility;
EOF

# Make the desktop shortcut executable
chmod +x ~/Desktop/negicc-station.desktop
```

Once installed, you can double-click the **NegICC Station** icon on your desktop to start the control panel window.

#### Log File Location
Stdout and stderr logs from the launcher and all components are appended to:
* `~/.config/negicc-station/log`

The central launcher window ([ui_main.py](src/ui_main.py)) acts as the control panel for the scanning station, providing quick access to the following three components:

1. **Crosstalk Calibration**: Calibrate RGB sensor crosstalk correction matrices using red, green, and blue negative color target exposures.
2. **Film Profiling**: Capture and read IT8 target patches, perform curve fitting, and compile custom film ICC profiles.
3. **Capture & Scan**: Manage tethered Sony A7R4 sessions over USB, trigger live single-shot or 4-shot IBIS captures, apply film negative inversions using loaded profiles, and save 16-bit linear TIFF files.

Alternatively, you can start any of the desktop application modules directly:
```bash
# Launch the film profiling UI directly
./venv/bin/python3 src/ui_film_profiling.py

# Launch the crosstalk calibration UI directly
./venv/bin/python3 src/ui_crosstalk_correction.py

# Launch the scanner capture UI directly
./venv/bin/python3 src/ui_capture.py
```


### CUDA Acceleration & Pipeline Benchmarking

#### 1. CUDA Compiler Environment Configuration
The [Makefile](Makefile) automatically searches for the NVIDIA CUDA compiler (`nvcc`) in your system executable path. On NVIDIA Jetson platforms, `nvcc` resides in `/usr/local/cuda/bin`. If detected, the build system automatically compiles the CUDA pipeline object (`build/color_conversion_cuda.o`), linking the GPU color conversion backend and setting `HAVE_CUDA=1` flags.

Ensure that the CUDA toolkit binary directory is added to your environment variables:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 2. Running the Performance Benchmark
To run the performance benchmark comparing processing times and verify parity across Python (NumPy), C++ CPU (Little CMS), and CUDA GPU pipelines:
```bash
make compare_pipelines
```
This target decompresses the reference RAW image `test_imgs/sample_portra400.ARW.xz`, processes it through all three pipeline modes, and displays:
* Parity deviation metrics (Maximum LSB difference and Mean LSB difference).
* Backend processing times in seconds.
* Speedup factors achieved by CUDA relative to CPU and NumPy.

---

## 4. Agent Instructions for Managing Dependencies

When introducing any new third-party dependency, library, or system package to this codebase, the agent **MUST** follow these protocol steps to keep the environment reproducible:

1. **Update System Dependencies**: Add any new system-level package requirements to the **Jetson Nano System Dependencies** section in this top-level [README.md](README.md).
2. **Setup Subdirectory Integration**: If the dependency is a third-party library, create a dedicated folder under `3rd_party/<DependencyName>/` and write a local `README.md` detailing how to download, compile, or install the library.
3. **Configure Git Exclusion**: If the dependency contains proprietary binaries or large compiled libraries, add them to the top-level [.gitignore](.gitignore) to prevent them from being checked into version control.
4. **Document Code & Builds**: Ensure all Makefiles and source files are updated and linked correctly, and document the complete build instructions so another agent can repeat the execution flow on a fresh Jetson Nano environment.

---

## 5. Troubleshooting and Hardware Diagnostics

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

## 6. Step-by-Step Film Profiling Guide

This guide details the complete protocol to calibrate a digital camera sensor's spectral crosstalk, capture film base characteristics, and profile target data at different exposures using the scanning software.

### Phase 1: Spectral Crosstalk Calibration
Before scanning negative film, you must capture the camera sensor's specific channel overlap characteristics under single-color narrow-band illuminations (or filter bands):

1. Launch the crosstalk calibration script:
   ```bash
   ./venv/bin/python3 src/ui_crosstalk_correction.py
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

---

## 7. Step-by-Step Scanning and Capture Guide (Capture & Scan)

This section details the workflow to perform tethered film negative scans, calibrate the film base orange mask, apply film-specific profiles, and export color-corrected 16-bit linear TIFFs.

### Phase 1: Initialize Session and Connect
1. Start the main launcher application:
   ```bash
   ./venv/bin/python3 src/ui_main.py
   ```
2. Click **Open Capture** to launch the Scanning & Capture panel (the main launcher window will automatically close). Alternatively, you can run `./venv/bin/python3 src/ui_capture.py` directly.
3. Confirm that the camera status indicator displays a green connected symbol:
   * `● Camera: Connected` (indicates successful communication with the Sony A7R4 over USB).
   * Orange (`●`) indicates a connection attempt is in progress.
   * Red (`●`) indicates disconnection.

### Phase 2: Calibrate the Film Base (Orange Mask Removal)
To clean and neutralize the film base's orange tint, you must record a reference profile of the unexposed emulsion:
1. Navigate to the **Film Base** tab in the central notebook.
2. Position a clear, unexposed but developed frame of negative film (e.g., from the film's orange leader strip) in the scanner gate.
3. Choose the capture parameters:
   * Select a manual shutter speed or check the **Auto Exposure** option.
4. Click the unified yellow **Capture Film Base** button in the left sidebar (the capture button is now a single dynamic button whose label and color adapt to the selected notebook tab).
5. Once the raw preview is displayed, drag a bounding box selection area over a clear portion of the film base.
6. Click **Read Film Base Values**. The average RGB readings will populate, updating the active calibration values. The tab label will switch to a green **Film Base** status.

### Phase 3: Negative Film Capture and Correction
1. Switch to the **Capture** tab.
2. Click **Load Profile...** to load a calibration profile JSON file. 
   * The application supports loading both full profiles (containing IT8 targets and custom ICC tables) and crosstalk-only calibration profiles (containing only the sensor's crosstalk matrix).
3. If the profile includes custom color calibration targets:
   * Select the appropriate target patch reference from the **Select Profile Target** list table displayed below the preview.
4. Select the **Capture Mode**:
   * **Single Shot Capture**: Standard high-resolution frame capture.
   * **Sony 4-Shot Pixel Shift**: Captures four consecutive raw frames with sub-pixel sensor shifting to reconstruct full RGB detail at every pixel site without Bayer interpolation.
5. Set the exposure:
   * Enable **Auto Exposure** or set a manual shutter speed. If Auto Exposure is enabled, the program runs an adaptive search, printing real-time per-channel dynamic range evaluation steps in the sidebar list.
6. Adjust the digital gain:
   * You can edit the numeric gain text field directly or click the `-` / `+` adjustment buttons.
   * When the gain text entry is not focused, you can also press the `+` / `-` keys on your keyboard to quickly modify the gain in increments of `0.10`.
7. Configure the crop orientation:
   * Use the rotation buttons (`0°`, `90°`, `180°`, `270°`) and toggles (`H-Flip`, `V-Flip`) to match the orientation of the physical negative.
8. Click the unified green **Capture Image** button in the left sidebar.
9. Analyze the captured image results:
   * **Histograms**: The right-hand sidebar displays the uncorrected RAW linear channel levels and the corrected, inverted positive preview histograms.
   * **Dynamic Range Display**: Shows the computed dynamic range per channel. If a channel overflows, the UI displays `Overexposed` for that channel rather than showing confusing raw negative values.

### Phase 4: Exporting the Linear TIFF
1. Once the preview looks correct, click the **Save TIFF...** button.
2. Choose the save location.
3. The application will convert the full-size raw capture using the active film base reference and profile correction, writing out a 16-bit linear RGB TIFF.
4. The orientation you specified is appended directly to the output file's EXIF metadata tags in-place, preserving your rotation preference without rewriting the raw pixel buffer.
5. In case of camera disconnects, hardware timeouts, or capture errors during the scan loop, the traceback and exception message will be logged cleanly to the console's standard output (`stdout`) instead of interrupting your session with a modal error dialog.
