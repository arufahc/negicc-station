# Film Profiling Guide

This document describes the step-by-step procedure for capturing film base calibration data and IT8 targets using the `sample_film_profiling.py` utility to prepare for generation of a 32-bit floating-point film stock inversion profile.

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
./venv/bin/python3 src/sample_film_profiling.py
```
Upon startup, the application will automatically attempt to connect to the Sony camera. The status indicator in the left sidebar will change to green: **Camera: Connected**.

---

## 2. Load Crosstalk Calibration

To apply real-time crosstalk correction:
1. Click the **LOAD CROSSTALK PROFILE** button in the left sidebar.
2. Select your camera's crosstalk calibration JSON file (containing the normalized correction matrices).
3. The sidebar status will display the loaded camera model, and the capture actions will become active.

---

## 3. Calibrate Film Base

Measuring the film base (unexposed developed film) is required to calibrate the scanner's light source response relative to the film's mask.

1. Insert an unexposed segment of the developed film stock into the scanner gate.
2. Navigate to the **Film Base** tab in the center panel.
3. Click the **CAPTURE FILM BASE** button in the left sidebar. The camera will tether-capture the frame, run crosstalk correction, and display the preview.
4. Using the mouse, click and drag a rectangular selection box over a uniform area of the film base image preview.
5. Click the **Read Film Base Values** button on the tab's toolbar.
6. The average linear red, green, and blue values along with their standard deviations are computed in 32-bit float space and displayed in the table below the preview.
7. The **Film Base** tab label will turn green, indicating that base calibration is complete.

---

## 4. Capture IT8 Targets

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

## 5. Saving the Session Profile

Once you have completed all target measurements and film base readings:
1. Click the **SAVE PROFILE** button in the left sidebar under *Profile Management*.
2. A prompt will appear asking for the **Film Stock Name** (e.g. `Portra400`, `Gold200`).
3. A save file dialog will open, default-named with the stock name and timestamp. Save the session profile to a JSON file.
4. The output JSON file will contain:
   - Top-level camera model name.
   - The original crosstalk correction profile.
   - The Film Base exposure settings (ISO, shutter speed) and float32 color values.
   - A list of all captured Target sessions, each containing its exposure info (ISO, shutter speed) and the float32 average and standard deviation measurements for every IT8 cell.

This JSON file is ready to be loaded by downstream profiling scripts to compile the final inversion mapping.

---

## 6. Building the Custom ICC Profile and Scanning Positive Previews

Once you have saved the film stock session profile JSON, you can use the `sample_build_prof.py` application to compile the final ICC profile and verify positive image rendering directly:

### Step 1: Start the Profile Builder App
Launch the application:
```bash
./venv/bin/python3 src/sample_build_prof.py
```

### Step 2: Download & Parse Reference IT8 File
1. In the sidebar under **1. Reference IT8 File**, specify the URL or local file path to the IT8 reference certificate. It supports both `.txt`/`.it8` raw reference files and `.zip` archives. (The default is `http://www.colorreference.de/targets/R190808.zip`).
2. Click **Download Reference IT8 File**. The app downloads the file, caches it inside the project's `data/` directory. If it is a ZIP archive, it lists the contents and prompts you to select which file to use. The app parses the selected `.txt`/`.it8` reference file, displays the actual loaded filename in the UI, and creates the corresponding JSON cache.

### Step 3: Load the Film Profile JSON
1. Under **2. Loaded Film Profile JSON**, click **Load Profile JSON...**.
2. Select the JSON file you saved in Step 5.
3. The app loads the profile, parses the crosstalk correction matrix and average film base RGB values, and fills the adjustment parameters.

### Step 4: Compile the Custom ICC Profile
1. Under **3. Generate Custom ICC Profile**, click **Compile Custom ICC Profile**.
2. The application performs the following steps automatically:
   - Generates a configuration header file `build_prof.h` containing your film's tone response curves (TRC) and crosstalk correction matrix.
   - Compiles `src/make_icc.c` using the C compiler and links it with `lcms2`.
   - Runs ArgyllCMS `colprof` to compute the raw color look-up table (cLUT) and matrix profiles.
   - Runs the compiled `make_icc` tool to merge the tone curves, crosstalk matrix, and Argyll's cLUT into a final custom color profile.
   - Saves the final profile to `profiles/<FilmName> cLUT.icc`.
   - Runs `profcheck` to evaluate error rates, displaying the average and maximum Delta E values directly in the sidebar.

### Step 5: Capture and Render Corrected Positive Previews
1. Check the **Apply IT8 & CC Corrections** checkbox under adjustments.
2. In the adjustment fields, you can tune the **Exposure Comp** and **Post Gamma** scaling. The **Custom Film Base** values default to the averages loaded from the film profile, but can be customized manually.
3. Set your target shutter speed and click **TETHERED CAPTURE PREVIEW**.
4. The Sony camera remote captures the RAW file, transfers it to the Nvidia Jetson Nano, and the updated C++ tethering library processes the pixel data (applying the film base ratios, crosstalk matrix, post gamma, and custom ICC profile conversion) in C++.
5. The display-ready positive sRGB image is returned directly to Python and drawn in the preview window!

---

## 7. Mathematical Profiling Steps

The profiling pipeline in `src/film_profiling.py` processes raw IT8 patch values ($R_{\text{raw}}, G_{\text{raw}}, B_{\text{raw}}$) through the following mathematical steps to compile the lookup table for the custom ICC profile:

### 1. Film Base Normalization (Transmittance Calculation)
To calculate absolute transmittance relative to the unexposed film base, the raw patch values are scaled by the average film base readings ($fb_r, fb_g, fb_b$), mapping the film base reference level to a target of `55000.0` (towards the upper range of the 16-bit scale range):

$$R_{\text{norm}} = R_{\text{raw}} \times \frac{55000.0}{fb_r}$$

$$G_{\text{norm}} = G_{\text{raw}} \times \frac{55000.0}{fb_g}$$

$$B_{\text{norm}} = B_{\text{raw}} \times \frac{55000.0}{fb_b}$$

This scales the patch responses such that a transmittance of $100\%$ (equal to the unexposed film mask) corresponds exactly to a value of $55000.0$ across all channels.

### 2. Grayscale Color Balancing (Neutral Alignment)
To ensure consistent neutrality (gray tracking) across different density levels, a specific grayscale patch (the "color balance cell") is selected via optimization:
* **Search:** The algorithm evaluates each grayscale patch $i \in [gs_0, gs_{23}]$ and simulates scaling the Green and Blue channel coefficients to match Red (forcing $R = G = B$ at patch $i$).
* **MSE Minimization:** It computes the Mean Squared Error (MSE) of all grayscale patches relative to their average channel values:
  
  $$\text{MSE} = \text{mean}\left( (R_{\text{bal}} - \text{avg})^2 + (G_{\text{bal}} - \text{avg})^2 + (B_{\text{bal}} - \text{avg})^2 \right)$$

* **Selection & Scaling:** The patch that minimizes this MSE is chosen as the optimal anchor (with normalized values $cb_r, cb_g, cb_b$). The final color-balanced values for all patches are calculated as:

  $$R_{\text{bal}} = R_{\text{norm}}$$

  $$G_{\text{bal}} = G_{\text{norm}} \times \frac{cb_r}{cb_g}$$

  $$B_{\text{bal}} = B_{\text{norm}} \times \frac{cb_r}{cb_b}$$

### 3. Preserving Transmittance (Bypassed Exposure Scaling)
To preserve the absolute transmittance values normalized to the film base, the global mid-grey exposure scaling step is bypassed:

$$S_{\text{global}} = 1.0$$

The final corrected patch values used to build the TI3 file for `colprof` are simply:

$$R_{\text{final}} = R_{\text{bal}}$$

$$G_{\text{final}} = G_{\text{bal}}$$

$$B_{\text{final}} = B_{\text{bal}}$$

