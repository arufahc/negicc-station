# Software Design Document

## Capture and Decoding: Obtaining Linear Values with LibRaw

This section details the software design, processing pipelines, and configuration necessary to capture RAW image data and decode it into mathematically precise **linear RGB values** using `LibRaw`. It covers both single-shot debayering and 4-shot pixel-shift reconstruction.

---

### 1. Overview and Core Objectives

When digitizing color negatives, the pixel values must represent physical transmittance (the fraction of light passing through the film) in a linear space. Non-linear conversions (such as camera-internal white balancing, gamma curves, auto-exposure, or auto-brightness) distort the relationship between light intensity and raw sensor response, making accurate negative inversion mathematically impossible.

To obtain true linear values:
1. The RAW data must be decoded without applying gamma curves ($\gamma = 1.0$).
2. No automatic scaling, white balance multipliers, or auto-brightness adjustments should be performed.
3. In single-shot mode, missing color components must be reconstructed via interpolation (**debayering**).
4. In pixel-shift mode, multiple exposures are merged to reconstruct a complete RGB triplet for every pixel location **without interpolation**, preserving film grain integrity.

---

### 2. Bayer Filter Array & LibRaw Sensor Extraction

Most digital cameras use a Single-Sensor Bayer Color Filter Array (CFA). A repeating $2\times2$ grid of color filters covers the sensor:

```
+---+---+
| R | G |
+---+---+
| G | B |
+---+---+
```

Each physical sensor pixel registers only one color channel (Red, Green, or Blue). When a RAW file (e.g., Sony `.ARW`) is loaded, LibRaw represents the sensor grid in `proc->imgdata.image`, where only one of the RGB components is filled for any given coordinate, while the other channels remain zero.

To obtain linear values, LibRaw parameters must be configured to bypass automatic post-processing.

#### Common LibRaw Parameters for Linear Output:
- `output_bps = 16`: Output 16 bits per channel to preserve the raw bit depth (usually 14-bit on high-end cameras).
- `user_flip = 0`: Disable automatic rotation to keep sensor-native coordinates.
- `gamm[0] = 1.0` and `gamm[1] = 1.0`: Force a linear gamma curve (identity mapping).
- `no_auto_bright = 1`: Disable automatic histogram-based highlight boosting.
- `no_auto_scale = 1`: Disable automatic exposure scaling.
- `highlight = 1`: Clip highlight values above saturation without attempting reconstruction.
- `output_color = 0`: Keep the camera-native color space (no conversion to sRGB or AdobeRGB).

---

### 3. Single-Shot Capture and Debayering (Interpolation)

In single-shot mode, LibRaw must interpolate the missing color channels for every pixel.

#### Processing Pipeline
1. Configure `LibRaw` parameters as described above.
2. In addition, disable camera-internal white balance scaling by setting:
   ```cpp
   proc->imgdata.params.use_auto_wb = 0;
   proc->imgdata.params.user_mul[0] = 1.0;
   proc->imgdata.params.user_mul[1] = 1.0;
   proc->imgdata.params.user_mul[2] = 1.0;
   proc->imgdata.params.user_mul[3] = 1.0;
   ```
3. Set the interpolation quality `user_qual`:
   - **Quality 0 (Bilinear)**: Highly recommended for film scanning. While advanced algorithms (e.g., AHD, VNG) produce sharper edges for natural scenes, they interpolate missing channels by looking at gradients in other channels. For negative film, this causes grain structure in the dense red channel to bleed into the green and blue channels, producing artificial grid-like grain artifacts. Bilinear interpolation performs simple averaging, preventing inter-channel grain bleeding.
4. Call `proc->dcraw_process()` to perform debayering and generate the linear 16-bit RGB image.

---

### 4. Sony 4-Shot Pixel-Shift Capture and Merging (No Interpolation)

This pixel-shift pipeline is designed specifically and exclusively for Sony 4-shot pixel-shift captures. The camera physical sensor shifts by exactly one pixel pitch between four successive exposures using the in-body image stabilization (IBIS) coils.

#### Sensor Displacement Pattern
The relative displacement $(x, y)$ of the sensor for each of the 4 shots is:
- **Shot 1**: $(0, 0)$ — Base frame.
- **Shot 2**: $(0, 1)$ — Shifted down by 1 pixel.
- **Shot 3**: $(-1, 1)$ — Shifted left by 1 pixel, down by 1 pixel.
- **Shot 4**: $(-1, 0)$ — Shifted left by 1 pixel.

By combining the 4 shots, every physical spot on the scene is sampled by all color filters:
- Two green filter samples.
- One red filter sample.
- One blue filter sample.

This allows us to assemble a complete RGB triplet for every pixel **without interpolation**.

#### Processing Pipeline
1. Load all 4 RAW files with `no_interpolation = 1`. This instructs LibRaw to bypass debayering:
   ```cpp
   proc->imgdata.params.no_interpolation = 1;
   proc->raw2image();
   proc->subtract_black();
   ```
   At this point, `proc->imgdata.image` contains the raw sensor matrix where each pixel has only a single channel populated.
2. Allocate a target image (the base frame `proc[0]`).
3. For each pixel coordinate `(r, c)` in the target image:
   - Identify the pixel displacements for each shot `mi`:
     - $dr = \text{movements}[mi][1]$
     - $dc = \text{movements}[mi][0]$
   - Retrieve the color channel index `col` active at `(r, c)` on the sensor using `proc[mi]->COLOR(r, c)`.
   - Copy the value from the shifted source to the target:
     - For **Red** (`col == 0`) and **Blue** (`col == 2`), copy directly:
       $$\text{Target}(r+dr, c+dc)[\text{col}] = \text{Source}_{mi}(r, c)[\text{col}]$$
     - For **Green** (`col == 1` or `col == 3`), accumulate and average the two samples from the exposures to reduce noise:
       $$\text{Target}(r+dr, c+dc)[1] = \frac{\text{Source}_{mi1}(r, c)[\text{col}] + \text{Source}_{mi2}(r, c)[\text{col}]}{2}$$
4. Re-label the output color count to 3 (`proc[0]->imgdata.idata.colors = 3`) to mark it as a full RGB image.

---

### 5. CPython C-Extension Interface

To allow high-performance integration with Python-based negative inversion and processing pipelines, the core capture and LibRaw decoding routines are exposed as a Python C-extension module named `negicc_station`.

Refer to the following source files for implementation details:
- [image_capture.h](src/image_capture.h): C++ CapturedImage interface representing camera parameters, captured raw file locations, and linear RGB/TIFF conversion routines.
- [image_capture.cpp](src/image_capture.cpp): Core implementation of single and pixel-shift linear conversions, including 2x2 downsampling.
- [python_bindings.cpp](src/python_bindings.cpp): CPython glue code exposing `negicc_station.capture()`, `negicc_station.CapturedImage`, and C-API NumPy array generation.
- [sample_ui.py](src/sample_ui.py): PyGObject/GTK3 desktop graphical interface demonstrating real-time camera controls, preview rendering, and timing diagnostics.

---

### 6. Auto-Exposure Search Algorithm & Overexposure Constraint

To automate the selection of the optimal shutter speed, the system integrates a hill-climbing search algorithm based on dynamic range maximization.

#### Objective Function
The algorithm evaluates exposure frames to maximize either:
- **`ALL` channels (default)**: Maximizes the average dynamic range across R, G, and B.
  $$\text{Objective} = \frac{\text{DR}_R + \text{DR}_G + \text{DR}_B}{3}$$
- **Individual channels (`R`, `G`, or `B`)**: Maximizes the dynamic range for the selected channel specifically.

Dynamic range ($\text{DR}_c$) is defined as the difference between the 95th and 5th percentile values in color channel $c$ within the active film area:
$$\text{DR}_c = P_{95}(\text{PixelValues}_c) - P_{5}(\text{PixelValues}_c)$$

To prevent clear light source bleeds or black film holder edges from throwing off the percentiles, the dynamic range is calculated only on a central cropped area, excluding a 5% border on all sides.

#### Overexposure Limit Constraint
For accurate film negative color inversion, no channel is allowed to reach or exceed sensor highlight saturation.
- High-end digital cameras typically utilize a 14-bit analog-to-digital converter (ADC), yielding a maximum raw capacity of 16384 levels.
- To prevent clipping and guarantee highlight headroom, the 95th percentile ($P_{95}$) of each channel is monitored and constrained to be below **80% of 16384 (13107.2)**.
- If $P_{95}$ for any channel exceeds 13107.2, that channel's dynamic range metric is heavily penalized:
  $$\text{Penalty} = 100000.0 + 10000.0 \times (P_{95} - 13107.2)$$
  $$\text{DR}_c = (P_{95} - P_5) - \text{Penalty}$$

This penalty function guarantees that any exposure where the 95th percentile exceeds the safety threshold is rejected in favor of a safe, unclipped exposure. Checking the 95th percentile instead of the absolute peak pixel value also makes the overexposure constraint robust against hot pixels and sensor noise.

---

### 7. Crosstalk Correction Mathematics & Principles

In film negative scanning systems, obtaining independent readings for each color channel (Red, Green, and Blue) is critical. However, even when utilizing narrow-band LED light sources or high-quality optical filters, digital camera sensors suffer from **spectral crosstalk**. Spectral crosstalk occurs because the transmission curves of the sensor's Color Filter Array (CFA) overlap (for example, the green filter has non-zero transmission in the red and blue bands). Consequently, a pure red illumination source will register non-zero responses in the green and blue channels of the raw linear image.

To mathematically decouple these overlapping signals, a crosstalk correction matrix is applied to the raw linear RGB response.

#### Mathematical Model

Let the raw linear RGB response of a pixel be represented by the vector:
$$V_{raw} = \begin{bmatrix} R_{raw} \\ G_{raw} \\ B_{raw} \end{bmatrix}$$

We define the corrected linear RGB response vector as:
$$V_{corr} = \begin{bmatrix} R_{corr} \\ G_{corr} \\ B_{corr} \end{bmatrix}$$

The corrected values are calculated via a linear transformation using a $3\times3$ correction matrix $C$:
$$V_{corr} = C \cdot V_{raw}$$

In expanded matrix form:
$$\begin{bmatrix} R_{corr} \\ G_{corr} \\ B_{corr} \end{bmatrix} = \begin{bmatrix} C_{00} & C_{01} & C_{02} \\ C_{10} & C_{11} & C_{12} \\ C_{20} & C_{21} & C_{22} \end{bmatrix} \begin{bmatrix} R_{raw} \\ G_{raw} \\ B_{raw} \end{bmatrix}$$

After matrix multiplication, the corrected channels are clipped to the 16-bit linear buffer limits $[0, 65535]$ to prevent underflow and overflow:
$$V_{corr, i} = \max\left(0, \min\left(65535, V_{corr, i}\right)\right) \quad \text{for } i \in \{0, 1, 2\}$$

---

### 8. Calibration and Matrix Generation

Calibration is performed to generate the correction matrix $C$ by measuring the sensor's specific crosstalk signature under controlled, single-channel illumination.

#### Calibration Capture Protocol

The calibration process requires capturing three separate exposures, each under a single narrow-band light source or bandpass filter:
1. **Red Calibration Capture**: Image exposed only with Red light.
2. **Green Calibration Capture**: Image exposed only with Green light.
3. **Blue Calibration Capture**: Image exposed only with Blue light.

For each of the three calibration images, the spatial average (mean) of the linear $R, G, B$ channels is calculated over a central region of interest. To avoid edge effects and lens vignetting, this central region is defined as a circle in the center of the image with a diameter equal to $1/3$ of the shorter side of the image.

This yields three average response vectors:
$$S_R = \begin{bmatrix} R_R \\ G_R \\ B_R \end{bmatrix}, \quad S_G = \begin{bmatrix} R_G \\ G_G \\ B_G \end{bmatrix}, \quad S_B = \begin{bmatrix} R_B \\ G_B \\ B_B \end{bmatrix}$$

#### Matrix Construction and Normalization

A raw crosstalk matrix $M$ is constructed where each column represents the response to one of the calibration lights:
$$M = \begin{bmatrix} S_R & S_G & S_B \end{bmatrix} = \begin{bmatrix} R_R & R_G & R_B \\ G_R & G_G & G_B \\ B_R & B_G & B_B \end{bmatrix}$$

To prevent overall brightness scaling from distorting the color balance, $M$ is normalized column-wise. Each column $j$ is divided by its diagonal element $M_{j,j}$ (the response of channel $j$ to its own corresponding illumination):
$$M_{norm} = \begin{bmatrix} 1 & \frac{R_G}{G_G} & \frac{R_B}{B_B} \\ \frac{G_R}{R_R} & 1 & \frac{G_B}{B_B} \\ \frac{B_R}{R_R} & \frac{B_G}{G_G} & 1 \end{bmatrix}$$

#### Correction Matrix Computation

The final crosstalk correction matrix $C$ is the mathematical inverse of the normalized crosstalk matrix:
$$C = M_{norm}^{-1}$$

If the matrix is singular (i.e., columns are linearly dependent due to severe sensor saturation or zero illumination), the inverse does not exist, and calibration fails.

> [!IMPORTANT]
> **Overexposure Constraint During Calibration:**
> The calibration exposures must maximize the dynamic range of each channel to optimize the signal-to-noise ratio, but **must strictly avoid overexposure (clipping)**. If any pixel values in the calibration region saturate (reach the 14-bit sensor ceiling of 16384), the response becomes non-linear, distorting the calculated ratios and rendering the calibration matrix invalid.


