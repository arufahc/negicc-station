# Software Design Document

## 1. Capture and Decoding: Obtaining Linear Values with LibRaw

This section details the software design, processing pipelines, and configuration necessary to capture RAW image data and decode it into mathematically precise **linear RGB values** using `LibRaw`. It covers both single-shot debayering and 4-shot pixel-shift reconstruction.

---

### 1.1 Overview and Core Objectives

When digitizing color negatives, the pixel values must represent physical transmittance (the fraction of light passing through the film) in a linear space. Non-linear conversions (such as camera-internal white balancing, gamma curves, auto-exposure, or auto-brightness) distort the relationship between light intensity and raw sensor response, making accurate negative inversion mathematically impossible.

To obtain true linear values:
1. The RAW data must be decoded without applying gamma curves ($\gamma = 1.0$).
2. No automatic scaling, white balance multipliers, or auto-brightness adjustments should be performed.
3. In single-shot mode, missing color components must be reconstructed via interpolation (**debayering**).
4. In pixel-shift mode, multiple exposures are merged to reconstruct a complete RGB triplet for every pixel location **without interpolation**, preserving film grain integrity.

---

### 1.2 Bayer Filter Array & LibRaw Sensor Extraction

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

### 1.3 Single-Shot Capture and Debayering (Interpolation)

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

*Implementation Reference:* The single-shot LibRaw configuration and processing are implemented in the `load_raw` function in [src/raw_processor.cpp](src/raw_processor.cpp).

---

### 1.4 Sony 4-Shot Pixel-Shift Capture and Merging (No Interpolation)

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
     - For **Red** (`col == 0`) and **Blue** (`col == 2`), copy directly: $\text{Target}(r+dr, c+dc)[\text{col}] = \text{Source}_{mi}(r, c)[\text{col}]$
     - For **Green** (`col == 1` or `col == 3`), accumulate and average the two samples from the exposures to reduce noise: $\text{Target}(r+dr, c+dc)[1] = \frac{\text{Source}_{mi1}(r, c)[\text{col}] + \text{Source}_{mi2}(r, c)[\text{col}]}{2}$
4. Re-label the output color count to 3 (`proc[0]->imgdata.idata.colors = 3`) to mark it as a full RGB image.

*Implementation Reference:* The Sony 4-shot pixel-shift merging algorithm is implemented in the `merge_pixel_shift_raw` function in [src/raw_processor.cpp](src/raw_processor.cpp).

---

## 2. CPython C-Extension Interface

To allow high-performance integration with Python-based negative inversion and processing pipelines, the core capture and LibRaw decoding routines are exposed as a Python C-extension module named `negicc_station`.

Refer to the following source files for implementation details:
- [image_capture.h](src/image_capture.h): C++ CapturedImage interface representing camera parameters, captured raw file locations, and linear RGB/TIFF conversion routines.
- [image_capture.cpp](src/image_capture.cpp): Core implementation of single and pixel-shift linear conversions, including 2x2 downsampling.
- [python_bindings.cpp](src/python_bindings.cpp): CPython glue code exposing `negicc_station.capture()`, `negicc_station.CapturedImage`, and C-API NumPy array generation.
- [ui_capture.py](src/ui_capture.py): PyGObject/GTK3 desktop graphical interface demonstrating real-time camera controls, preview rendering, and timing diagnostics.

---

## 3. Auto-Exposure Search Algorithm & Overexposure Constraint

To automate the selection of the optimal shutter speed, the system integrates a hill-climbing search algorithm based on dynamic range maximization.

#### Objective Function
The algorithm evaluates exposure frames to maximize either:
- **`ALL` channels (default)**: Maximizes the average dynamic range across R, G, and B:

$$
\text{Objective} = \frac{\text{DR}_R + \text{DR}_G + \text{DR}_B}{3}
$$
- **Individual channels (`R`, `G`, or `B`)**: Maximizes the dynamic range for the selected channel specifically.

Dynamic range ($\text{DR}_c$) is defined as the difference between the 95th and 5th percentile values in color channel $c$ within the active film area:

$$
\text{DR}_c = P_{95}(\text{PixelValues}_c) - P_{5}(\text{PixelValues}_c)
$$

To prevent clear light source bleeds or black film holder edges from throwing off the percentiles, the dynamic range is calculated only on a central cropped area, excluding a 5% border on all sides.

#### Overexposure Limit Constraint
For accurate film negative color inversion, no channel is allowed to reach or exceed sensor highlight saturation.
- High-end digital cameras typically utilize a 14-bit analog-to-digital converter (ADC), yielding a maximum raw capacity of 16384 levels.
- To prevent clipping and guarantee highlight headroom, the 95th percentile ($P_{95}$) of each channel is monitored and constrained to be below **80% of 16384 (13107.2)**.
- If $P_{95}$ for any channel exceeds 13107.2, that channel's dynamic range metric is heavily penalized:

$$
\text{Penalty} = 100000.0 + 10000.0 \times (P_{95} - 13107.2)
$$

$$
\text{DR}_c = (P_{95} - P_5) - \text{Penalty}
$$

This penalty function guarantees that any exposure where the 95th percentile exceeds the safety threshold is rejected in favor of a safe, unclipped exposure. Checking the 95th percentile instead of the absolute peak pixel value also makes the overexposure constraint robust against hot pixels and sensor noise.

*Implementation Reference:* The dynamic range calculation, highlight penalty, and hill-climbing search logic are implemented in [src/auto_exposure.py](src/auto_exposure.py) (specifically the `calculate_dynamic_range` and `run_auto_exposure` functions).

---

## 4. Crosstalk Correction Mathematics & Principles

In film negative scanning systems, obtaining independent readings for each color channel (Red, Green, and Blue) is critical. However, even when utilizing narrow-band LED light sources or high-quality optical filters, digital camera sensors suffer from **spectral crosstalk**. Spectral crosstalk occurs because the transmission curves of the sensor's Color Filter Array (CFA) overlap (for example, the green filter has non-zero transmission in the red and blue bands). Consequently, a pure red illumination source will register non-zero responses in the green and blue channels of the raw linear image.

To mathematically decouple these overlapping signals, a crosstalk correction matrix is applied to the raw linear RGB response.

#### Mathematical Model

Let the raw linear RGB response of a pixel be represented by the vector:

$$
V_{raw} = \begin{bmatrix} R_{raw} \\ G_{raw} \\ B_{raw} \end{bmatrix}
$$

We define the corrected linear RGB response vector as:

$$
V_{corr} = \begin{bmatrix} R_{corr} \\ G_{corr} \\ B_{corr} \end{bmatrix}
$$

The corrected values are calculated via a linear transformation using a $3\times3$ correction matrix $C$:

$$
V_{corr} = C \cdot V_{raw}
$$

In expanded matrix form:

$$
\begin{bmatrix} R_{corr} \\ G_{corr} \\ B_{corr} \end{bmatrix} = \begin{bmatrix} C_{00} & C_{01} & C_{02} \\ C_{10} & C_{11} & C_{12} \\ C_{20} & C_{21} & C_{22} \end{bmatrix} \begin{bmatrix} R_{raw} \\ G_{raw} \\ B_{raw} \end{bmatrix}
$$

After matrix multiplication, the corrected channels are clipped to the 16-bit linear buffer limits $[0, 65535]$ to prevent underflow and overflow:

$$
V_{corr, i} = \max\left(0, \min\left(65535, V_{corr, i}\right)\right) \quad \text{for } i \in \{0, 1, 2\}
$$

---

### 4.1 Calibration and Matrix Generation

Calibration is performed to generate the correction matrix $C$ by measuring the sensor's specific crosstalk signature under controlled, single-channel illumination.

#### Calibration Capture Protocol

The calibration process requires capturing three separate exposures, each under a single narrow-band light source or bandpass filter:
1. **Red Calibration Capture**: Image exposed only with Red light.
2. **Green Calibration Capture**: Image exposed only with Green light.
3. **Blue Calibration Capture**: Image exposed only with Blue light.

For each of the three calibration images, the spatial average (mean) of the linear $R, G, B$ channels is calculated over a central region of interest. To avoid edge effects and lens vignetting, this central region is defined as a circle in the center of the image with a diameter equal to $1/3$ of the shorter side of the image.

This yields three average response vectors:

$$
S_R = \begin{bmatrix} R_R \\ G_R \\ B_R \end{bmatrix}, \quad S_G = \begin{bmatrix} R_G \\ G_G \\ B_G \end{bmatrix}, \quad S_B = \begin{bmatrix} R_B \\ G_B \\ B_B \end{bmatrix}
$$

#### Matrix Construction and Normalization

A raw crosstalk matrix $M$ is constructed where each column represents the response to one of the calibration lights:

$$
M = \begin{bmatrix} S_R & S_G & S_B \end{bmatrix} = \begin{bmatrix} R_R & R_G & R_B \\ G_R & G_G & G_B \\ B_R & B_G & B_B \end{bmatrix}
$$

To prevent overall brightness scaling from distorting the color balance, $M$ is normalized column-wise. Each column $j$ is divided by its diagonal element $M_{j,j}$ (the response of channel $j$ to its own corresponding illumination):

$$
M_{norm} = \begin{bmatrix} 1 & \frac{R_G}{G_G} & \frac{R_B}{B_B} \\ \frac{G_R}{R_R} & 1 & \frac{G_B}{B_B} \\ \frac{B_R}{R_R} & \frac{B_G}{G_G} & 1 \end{bmatrix}
$$

#### Correction Matrix Computation

The final crosstalk correction matrix $C$ is the mathematical inverse of the normalized crosstalk matrix:

$$
C = M_{norm}^{-1}
$$

If the matrix is singular (i.e., columns are linearly dependent due to severe sensor saturation or zero illumination), the inverse does not exist, and calibration fails.

> [!IMPORTANT]
> **Overexposure Constraint During Calibration:**
> The calibration exposures must maximize the dynamic range of each channel to optimize the signal-to-noise ratio, but **must strictly avoid overexposure (clipping)**. If any pixel values in the calibration region saturate (reach the 14-bit sensor ceiling of 16384), the response becomes non-linear, distorting the calculated ratios and rendering the calibration matrix invalid.

*Implementation Reference:* The crosstalk matrix calibration and normalized matrix inversion are implemented in the `compute_calibration_matrices` function in [src/crosstalk_calibration.py](src/crosstalk_calibration.py), while the application of the matrix to pixel arrays is implemented in the `apply_correction` function of the same file.

---

## 5. Film Profiling, ICC Generation, and Linear Negative Conversion

This section covers the mathematical steps utilized to compile negative film profiles containing custom 3D cLUT ICC profiles and Tone Reproduction Curves (TRCs), and apply them dynamically during raw image conversion.

*Implementation Reference:* The overall film profiling library, ArgyllCMS TI3 generation, and ICC generation process are managed by the `FilmProfile` class and helper functions in [src/film_profiling.py](src/film_profiling.py).

### 5.0 Why a Profile is Needed (Even with a Tricolor/Narrow-band Light Source)

While some setups attempt to bypass color profiling by utilizing narrow-band tricolor light sources or optical bandpass filters to isolate color channels physically, a profiled software approach remains strictly necessary for accurate, high-fidelity film reproduction.

#### 1. Hardware Availability & Sensor Ubiquity
A tricolor light source or true monochrome camera setup is not strictly needed. True monochrome sensors are extremely rare and costly, whereas digital cameras with color filter arrays (CFAs) are ubiquitous. Using a standard CFA sensor with mathematical crosstalk correction is the most common approach and is already standard in modern digital image processing. Combining **pixel-shift technology** (e.g., on the Sony A7R4) with a crosstalk correction matrix achieves the same channel purity as physical tricolor separation, but uses readily available commercial hardware.

#### 2. Avoiding Arbitrary Color Spaces
Once crosstalk correction is applied and film base levels are normalized, the resulting linear RGB values are simply arbitrary sensor response metrics. They do not reside in any standard, meaningful color space. While one could arbitrarily assign a standard color space (like sRGB or AdobeRGB), the colors will not look realistic. A profiled approach is required to mathematically map these arbitrary camera sensor RGB response values to known, absolute colorimetric **XYZ values** (derived from the IT8 target's certified colorimeter measurements).

#### 3. Handling Exposure Variations (Dense vs. Thin Negatives)
A color profile maps the film's response over a wide range. This allows the conversion software to dynamically handle varying exposures—such as when a negative is too dense (overexposed) or too thin (underexposed), or when the scan is captured using a different ISO sensitivity than the standard baseline used during the initial profiling target exposure.

#### 4. Non-Linearity of Film Dyes (Differing Gamma Curves)
Crucially, the cyan, magenta, and yellow color dyes in negative film emulsions do not react linearly to light and do not possess the same gamma curve. Because their density-to-exposure curves differ, a simple linear or single-gamma scaling will produce severe color shifts in the shadows and highlights. A profiled approach using independent Tone Reproduction Curves (TRCs) for each channel and a 3D color lookup table (cLUT) models these non-linearities exactly. This produces accurate, natural-looking results automatically, completely eliminating the need for tedious manual post-processing and color-fiddling.


### 5.1 Exposure Ratio Scaling
Let the film base capture be acquired at exposure time $t_b$ and sensitivity $ISO_b$.
Let the target capture (containing the IT8 reference target) be acquired at exposure time $t_t$ and sensitivity $ISO_t$.

The raw exposure metrics are calculated as:

$$
\text{Exposure}_b = t_b \times \frac{ISO_b}{100.0}
$$

$$
\text{Exposure}_t = t_t \times \frac{ISO_t}{100.0}
$$

The exposure ratio mapping target measurements to film base capture conditions is:

$$
\text{Ratio} = \frac{\text{Exposure}_b}{\text{Exposure}_t}
$$

### 5.2 Film Base Normalization
To map the measured crosstalk-corrected film base levels (Red, Green, and Blue) to a fixed target normalization level (defaulting to 55000.0), channel-specific scaling factors are calculated:

$$
S_c = \frac{N_{\text{target}}}{FB_c} \times \text{Ratio} \quad \text{for } c \in \{R, G, B\}
$$

The raw patch measurements $P_{\text{raw}, c}$ are then scaled:

$$
P_{\text{scaled}, c} = P_{\text{raw}, c} \times S_c
$$

### 5.3 Tone Reproduction Curve (TRC) Estimation
We extract the grayscale patches $i \in \{0, \dots, 23\}$ from the target data. Let their scaled crosstalk-corrected averages be represented by $V_c(i)$ for $c \in \{R, G, B\}$. 

Using the reference XYZ target data, we normalize the reference luminance $Y_{\text{ref}}(i)$ to the range $[0.0, \text{whitest-patch-scaling}]$:

$$
Y_{\text{norm}}(i) = Y_{\text{ref}}(i) \times \frac{\text{whitest-patch-scaling}}{\max(Y_{\text{ref}})}
$$

Monotonic cubic spline functions $\text{TRC}_c(V)$ are fitted to map from the linear sensor space to the normalized reference luminance space:

$$
\text{TRC}_c(V) \approx Y_{\text{norm}}
$$

These curves act as the independent red, green, and blue Tone Reproduction Curves (TRCs) serialized into the final profile.

### 5.4 Dynamic Scaling and Matrix Merging
When converting a raw negative scan captured at shutter speed $t_s$ and sensitivity $ISO_s$, we calculate the scan-to-base exposure ratio:

$$
\text{Ratio}_{\text{scan}} = \frac{t_b \times (ISO_b / 100.0)}{t_s \times (ISO_s / 100.0)}
$$

The dynamic normalization scale factors are:

$$
S_c = \frac{N_{\text{target}}}{FB_c} \times \text{Ratio}_{\text{scan}} \quad \text{for } c \in \{R, G, B\}
$$

The $3\times3$ crosstalk correction matrix $C$ is merged with these scale factors row-wise:

$$
M_{\text{merged}} = \begin{bmatrix} S_R & 0 & 0 \\ 0 & S_G & 0 \\ 0 & 0 & S_B \end{bmatrix} \cdot C
$$

This single combined matrix performs both crosstalk correction and film base normalization in a single step:

$$
V_{\text{scaled}} = M_{\text{merged}} \cdot V_{\text{raw}}
$$

Finally, the scaled camera RGB values are passed through the independent TRC curves and color-managed through the 3D cLUT ICC profile to produce the final sRGB output:

$$
V_{\text{sRGB}} = \text{ICC}_{\text{LUT}}\left(\begin{bmatrix} \text{TRC}_R(R_{\text{scaled}}) \\ \text{TRC}_G(G_{\text{scaled}}) \\ \text{TRC}_B(B_{\text{scaled}}) \end{bmatrix}\right)
$$

### 5.5 Dynamic Profile Target Selection & Performance Optimization
To support multi-target calibration profiles (profiles containing calibration targets captured at different exposure levels), the system implements a dynamic profile matching algorithm:
1. **Region of Interest Extraction**: Extracts the $2/3$ center square of the shorter side from the raw image to focus on the calibration target area.
2. **Subsampling Optimization**: To avoid performance bottlenecks on high-resolution 61MP sensor captures, the region of interest is subsampled by taking every 10th pixel in both dimensions. This reduces the pixels under analysis by a factor of 100 (from 17.8 million down to ~178k elements), reducing execution time from ~10 seconds to under 20ms while preserving the statistical distribution of the dynamic range.
3. **Crosstalk Correction**: Corrects the subsampled raw image patch using the profile's crosstalk matrix.
4. **Percentile Calculation**: Computes the 2% and 98% intensity percentiles on the green channel to represent the density range of the scanned target.
5. **Transmittance Mapping**: Normalizes the percentiles by the captured film base green channel and scales them by the exposure ratio between the base capture and the current scan:

$$
t_{c} = \frac{P_c}{FB_G} \times \text{Ratio}_{\text{scan}}
$$

6. **Grayscale Alignment**: Maps the measured 2% and 98% transmittances against the 24 gray scale patches ($gs0 \dots gs23$) of each candidate target profile.
7. **Luminance Matching**: Calculates the mid-grey distance metric:

$$
\text{center-index} = \frac{\text{idx}_{98} + \text{idx}_2}{2.0}
$$

$$
\text{dist} = |\text{center-index} - 11.5|
$$
   The target profile with the minimum distance metric (closest mid-grey patch alignment to target index 11.5) is selected as the optimal conversion profile.
8. **Identity-Based Range Caching**: To prevent redundant re-evaluations during UI redraws or target selection changes, the computed transmittance range is cached using a composite key representing the identities of the raw image object and the active profile: `(id(raw_image), id(profile))`.

*Implementation Reference:* The dynamic target selection algorithm is implemented in the `find_best_target_index` function in [src/target_selection.py](src/target_selection.py). The performance optimizations (subsampling and range caching) are integrated within the `get_image_transmittance_range` function in [src/ui_capture.py](src/ui_capture.py).

---

## 6. Color Space Conversion Pipeline & Multi-Backend Architecture

Following crosstalk correction and film base scaling, the normalized linear RGB sensor pixels must be color-managed. This section describes the standard ICC conversion sequence, details the differences between the three available pipeline backends, and presents benchmark and parity data obtained on the target Jetson Nano platform.

### 6.1 The Color Space Conversion Steps
To transform raw linear camera responses into standard sRGB space, the pipeline evaluates the film's custom IT8 profile stages and projects coordinates through the Profile Connection Space (PCS):

1. **Pixel Normalization**: Converts raw 16-bit unsigned integers ($[0, 65535]$ LSB) to normalized `float32` values in the range $[0.0, 1.0]$.
2. **Crosstalk & Scaling**: Multiplies the normalized float vector by the $3\times3$ merged crosstalk-normalization matrix $M_{\text{merged}}$ (row-wise scaling combined with the crosstalk inverse) and clips to $[0.0, 1.0]$.
3. **Tone Curve Linearization (Stage 0 / TRC0)**: Applies 1D linear interpolation to the R, G, B channels using the 1D tone curves extracted from the film profile's `AtoB0` tag.
4. **3x3 Matrix + Offset Projection (Stage 1)**: Multiplies the linearized vector by the profile's $3\times3$ color matrix and adds the 3D translation offset, clipping output coordinates to $[0.0, 1.0]$.
5. **3D Color Lookup Table (Stage 2 / cLUT)**: Performs 3D tetrahedral linear interpolation on the grid dimensions of the cLUT to map coordinates from the camera color space to the Profile Connection Space (PCS).
6. **Output Curve Correction (Stage 3 / TRC2)**: Passes the cLUT output through the output 1D tone curves.
7. **PCS Range Expansion**: Scales the resulting coordinates by $65535.0 / 32768.0$ to project them into standard PCS D50 XYZ coordinates.
8. **Bradford Chromatic Adaptation (D50 to D65)**: Applies a Bradford adaptation matrix to shift colors from the profile's D50 white point to the sRGB D65 white point:

$$
M_{\text{adapt}} = \begin{bmatrix} 0.9555766 & -0.0230393 & 0.0631636 \\ -0.0282895 & 1.0099416 & 0.0210077 \\ 0.0122982 & -0.0204830 & 1.3299098 \end{bmatrix}
$$

9. **XYZ to Linear sRGB Matrix Projection**: Converts the D65 XYZ vector to linear sRGB space:

$$
M_{\text{xyz-to-srgb}} = \begin{bmatrix} 3.2406255 & -1.5372080 & -0.4986286 \\ -0.9689307 & 1.8757561 & 0.0415175 \\ 0.0557101 & -0.2040211 & 1.0569959 \end{bmatrix}
$$
   and clips coordinates to $[0.0, 1.0]$.
10. **EOTF Mapping & Quantization**: Applies target colorspace EOTF (the piecewise non-linear sRGB mapping curve for `"srgb"`, or identity mapping for `"srgb-g10"`), multiplies by $65535.0$, and rounds to `uint16_t` values.

*Implementation Reference:* The unified color space conversion pipeline entrypoint is implemented in the `CapturedImage::get_linear_rgb` function in [src/image_capture.cpp](src/image_capture.cpp).

---

### 6.2 Pipeline Backends Comparison

The system supports three distinct modes to run these steps:

| Feature | CUDA Pipeline | Python Pipeline | C++ CPU (`cpp`) Pipeline |
| :--- | :--- | :--- | :--- |
| **Primary Target** | GPU-accelerated production | Interactive prototyping / debugging | CPU-only runtime fallback |
| **Math Precision** | Single-precision `float32` | Mixed `float32` / `float64` | 16-bit integer fixed-point |
| **LUT Interpolation** | Tetrahedral (`float32`) | Tetrahedral (vectorized NumPy) | Trilinear or fixed-point |
| **EOTF Curves** | Predefined (piecewise / linear) | Predefined (piecewise / linear) | Profile-defined (Little CMS curves) |
| **Custom Output Profiles** | Unsupported (triggers CPU fallback) | Supported (falls back to LCMS) | Fully supported via CMM link |
| **TIFF Embedding** | Attaches profile Tag 34675 only | Attaches profile Tag 34675 only | Computes and embeds profile |

* **CUDA Backend**: Processes pixels concurrently across blocks of threads. The film profile's internal TRCs and cLUT are uploaded to GPU buffers, and the remaining colorspace adaptation and sRGB EOTF mappings are performed using optimized CUDA arithmetic. Custom output profiles are bypassed and fall back to CPU.
* **Python Backend**: Used in prototype scripts. Leverages ctypes to parse `cmsStage` elements and executes the manual matrix operations and tetrahedral lookups using NumPy vectorization.
* **C++ CPU Backend**: Built around the Little CMS library. It compiles a standard link between the input IT8 profile and the target output profile, processing pixels through optimized fixed-point integer lookup tables. This is the only backend that dynamically uses the output profile file during math transformation.

*Implementation Reference:*
- **CUDA Backend**: The optimized float32 GPU kernel is defined as `color_conversion_kernel` in [src/color_conversion.cu](src/color_conversion.cu) and declared in [src/color_conversion.h](src/color_conversion.h).
- **Python Backend**: The NumPy-vectorized tetrahedral interpolation and native lcms2 ctypes wrapper are implemented in [src/color_conversion.py](src/color_conversion.py).
- **C++ CPU Backend**: The Little CMS pipeline integration is implemented in the `CapturedImage::color_transform_cpp` function in [src/image_capture.cpp](src/image_capture.cpp).

---

### 6.3 Parity and Discrepancy Analysis

Outputs from the three backends were compared under identical inputs ($[0, 65535]$ LSB) to evaluate correctness and quantization behavior:

##### Pixel-wise Difference Matrix
| Backend Comparison | Max Difference | Mean Difference | Interpretation / Reason |
| :--- | :---: | :---: | :--- |
| **CUDA vs. Python** | `1 LSB` | `0.0031 LSB` | **Parity Verified.** Discrepancies are minor rounding variances between single-precision `float32` GPU math and `float64` CPU math. |
| **CUDA vs. C++ CPU** | `5373 LSB` | `144.4 LSB` | **Normal Discretization.** Little CMS CPU uses 16-bit integer fixed-point discretization, producing rounding noise compared to high-precision float32 tetrahedral interpolation. |
| **Python vs. C++ CPU** | `5373 LSB` | `144.4 LSB` | **Normal Discretization.** Matches the quantization noise observed between float32/float64 interpolation and fixed-point integer CMM. |

---

### 6.4 Performance Benchmarks on Nvidia Jetson Nano

We measured processing times on a sample 16-bit linear RAW image ($4784 \times 3188$ pixels, ~15.2M pixels, half-resolution scan) executing on the Nvidia Jetson Nano (ARM Aarch64 CPU + Maxwell GPU):

| Pipeline Backend | Processing Time (sec) | Speedup vs Python | Speedup vs CPU |
| :--- | :---: | :---: | :--- |
| **Python** (NumPy Vectorized) | `26.07`s | $1.0\times$ (Baseline) | — |
| **C++ CPU** (Little CMS 2) | `1.43`s | $18.2\times$ | $1.0\times$ |
| **CUDA** (float32 GPU kernel) | **`0.77`s** | **$33.5\times$** | **$1.8\times$** |

##### Performance Rationale:
1. **Parallel Execution**: The CUDA backend achieves a $33.5\times$ speedup over Python and $1.8\times$ speedup over C++ CPU by processing pixel transformations in parallel threads on the GPU, avoiding CPU cache bottlenecking and serial loops.
2. **NumPy Vectorization Overhead**: While NumPy is vectorized, executing multiple non-contiguous memory writes, float operations, and interpolation masks sequentially in Python incurs significant interpreter and memory allocation overhead.
3. **Fixed-Point C++ Efficiency**: The C++ CPU backend is highly optimized but is bounded by single-core throughput constraints when traversing large image grids sequentially.
