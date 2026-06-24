# Film Profiling Guide

This document describes the step-by-step procedure for capturing film base calibration data and IT8 targets using the `ui_film_profiling.py` utility to prepare for generation of a 32-bit floating-point film stock inversion profile.

---

## Prerequisites

Before starting the profiling session, ensure that:
1. The tethered Sony A7R4 camera is connected over USB and recognized.
2. A crosstalk calibration profile (`.json`) for the camera has been previously generated and is ready to load.
3. The film scanning light source is powered on and stabilized.

---

## 1. Starting the Application

Launch the desktop profiling application:
```bash
./venv/bin/python3 src/ui_film_profiling.py
```
Upon startup, the application will automatically attempt to connect to the Sony camera. The status indicator in the left sidebar will change to green: **Camera: Connected**.

---

## 2. Load Crosstalk Calibration

To apply real-time crosstalk correction:
1. Click the **LOAD CROSSTALK PROFILE** button in the left sidebar.
2. Select your camera's crosstalk calibration JSON file (containing the normalized correction matrices).
3. The sidebar status will display the loaded camera model, and the capture actions will become active.

---

## 3. Load/Download IT8 Reference target

To compute accurate color conversions and verify profile quality:
1. In the sidebar under **IT8 Target Reference**, specify the URL or local file path to the IT8 reference certificate. It supports both `.txt`/`.it8` raw reference files and `.zip` archives. (The default is `http://www.colorreference.de/targets/R190808.zip`).
2. Click **DOWNLOAD REFERENCE IT8**. The app downloads the file, caches it inside the project's `data/` directory. If it is a ZIP archive, it lists the contents and prompts you to select which file to use. The app parses the selected `.txt`/`.it8` reference file, displays the actual loaded filename in the UI, and creates the corresponding JSON cache.

---

## 4. Calibrate Film Base

Measuring the film base (unexposed developed film) is required to calibrate the scanner's light source response relative to the film's mask.

1. Insert an unexposed segment of the developed film stock into the scanner gate.
2. Navigate to the **Film Base** tab in the center panel.
3. Click the **CAPTURE FILM BASE** button in the left sidebar. The camera will tether-capture the frame, run crosstalk correction, and display the preview.
4. Using the mouse, click and drag a rectangular selection box over a uniform area of the film base image preview.
5. Click the **Read Film Base Values** button on the tab's toolbar.
6. The average linear red, green, and blue values along with their standard deviations are computed in 32-bit float space and displayed in the table below the preview.
7. The **Film Base** tab label will turn green, indicating that base calibration is complete.

---

## 5. Capture IT8 Targets

Next, capture one or more IT8 target slides under different exposure conditions or color targets to sample film dye responses.

### Capture and Alignment Workflow:
1. Insert the IT8 target slide into the scanner gate.
2. Select the first **Target** tab (labeled `Target 1`).
3. Click the **CAPTURE TARGET** button in the left sidebar. The app will tether-capture, apply crosstalk correction in 32-bit float space, and render the preview.
4. Click the **Layer IT8 Mask** button on the tab's toolbar. A green grid representing the IT8 patch layout will overlay on the image preview.
5. Align the grid over the target slide's color patches:
   - Use the **Arrow keys** (Up, Down, Left, Right) to translate the grid position.
   - Use the **`+` (equals)** and **`-` (minus)** keys on the keyboard to scale the grid size.
   - Ensure that the green boxes are centered over the corresponding color cells without overlapping the black boundaries.
6. Click the **Read Mask Values** button.
   - The application calculates the average R, G, B channels and standard deviations in 32-bit float space for all 200+ patches.
   - The measurements are output to stdout and displayed in a copyable dialog.
   - The tab label text turns green to indicate the data is captured.

### Capturing Multiple Targets:
If you need to capture multiple targets (e.g. for multi-exposure blending or profiling multiple targets):
1. Click the **`+`** tab next to your current target tabs. A new, blank target tab (e.g., `Target 2`) is automatically created and activated.
2. Repeat the target capture, alignment, and read workflow for this new tab.
3. You can scroll through tabs horizontally using the horizontal scroll buttons if the number of tabs exceeds the panel width.

---

## 6. Saving the Profile and Generating Custom ICC Profile

Once you have completed all target measurements and film base readings:
1. Click the **SAVE PROFILE** button in the left sidebar under *Profile Management*.
2. A prompt will appear asking for the **Film Stock Name** (e.g. `Portra400`, `Gold200`).
3. A save file dialog will open, default-named with the stock name and timestamp. Save the session profile to a JSON file.
4. In the background, the application automatically performs the following compilation steps:
   - Computes tone response curves (TRC) and crosstalk correction matrices.
   - Compiles the custom C++ profiling utilities.
   - Invokes ArgyllCMS `colprof` to compute the raw color lookup table (cLUT).
   - Generates the final self-contained ICC profile and embeds it directly into the saved JSON profile.
5. Once complete, a **Film Profiling Report** window automatically opens to verify the profile quality:
   - **Tone Response Curves (TRC)**: Shows the measured TRC curves for Red, Green, and Blue.
   - **Characteristic Curve**: Compares the measured RGB values against the reference Y values.
   - **IT8 Profile Quality (profcheck)**: Shows the Delta E color accuracy stats.
   - **Target Converted**: Displays a real-time, inverted positive rendering of the captured IT8 targets using the compiled custom ICC profile.
   - **Compilation Logs**: Provides detailed output logs of the profile generation process.
