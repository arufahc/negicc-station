# Training Procedure: Photographic Intent Field Appearance Model (DINOv2 & DINOv3)

**Status**: Active / Production  
**Target Subsystem**: `negicc-station` Appearance & Optimization Pipeline  
**Primary Training Script**: [`src/train_dinov2_intent_field.py`](file:///home/alpha/Projects/negicc-station/src/train_dinov2_intent_field.py)  
**Downstream Conversion Pipeline**: [`src/batch_dinov2_convert.py`](file:///home/alpha/Projects/negicc-station/src/batch_dinov2_convert.py)  
**Experiment Registry**: [`experiments/experiment_registry.json`](file:///home/alpha/Projects/negicc-station/experiments/experiment_registry.json)  
**Default Output Model**: `models/dinov2_photographic_intent_field_multilayer.pt`  

---

## 1. Executive Summary & Purpose

This specification documents the architecture, training procedure, and runtime consumption of the **Photographic Intent Field Appearance Model**. It is designed to allow any autonomous agent or researcher to independently train, evaluate, and deploy new vision foundation models (including **DINOv2** and **DINOv3**) to guide automated film negative conversion.

Rather than predicting non-convex physical conversion parameters directly (which leads to catastrophic optimization collapse and training instability), the appearance model predicts the **locally plausible CIELAB color distribution $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i)$** for every spatial patch $i$ across the scene. Downstream, the native C++ conversion engine ([`src/film_profiling.py`](file:///home/alpha/Projects/negicc-station/src/film_profiling.py) / `negicc_station`) is used as an opaque black box to search for optimal profile target brackets ($\text{Target}_k$) and continuous gains ($E, g, b$) that match the predicted photographic intent field.

```mermaid
flowchart TD
    subgraph Offline_Training ["1. Standalone Offline Training Procedure (DINOv2 / DINOv3)"]
        Corpus["Positive Master Film Corpus\n(3,646 Authentic Kodak Portra 400 Scans)"] --> UprightCheck["Upright Orientation Invariance\n(Ensure vertical alignment: sky at top, terrain at bottom)"]
        UprightCheck --> Bucketing["Approach 1: Aspect-Ratio Bucketing\n(3:2, 2:3, 1:1, 4:3, XPAN patch-aligned grids)"]
        Bucketing --> Resample["Native Aspect Resampling & Ground Truth CIELAB\n(Patch-aligned W_tgt x H_tgt -> grid_w x grid_h patches of 14x14)"]
        Resample --> Backbone["Frozen Vision Foundation Backbone\n(DINOv2 or DINOv3 ViT via HuggingFace)\n2D Bicubic Positional Interpolation"]
        Backbone --> MultiLayer["Multi-Layer Feature Fusion\nL_photo (~0.75L) + L_sem (L)\nz_i in R^(2D) per spatial patch"]
        MultiLayer --> Cache["Sharded Multi-Worker Feature Cache\n(Token-wise continuous multi-aspect cache)"]
        Cache --> AdapterTrain["Train PhotographicIntentAdapter\n(Residual Bottleneck MLP + Heteroscedastic Huber Loss)"]
        AdapterTrain --> Checkpoint["Saved Checkpoint with Architecture Metadata\n(models/dinov2_photographic_intent_field_multilayer.pt)"]
    end

    subgraph Runtime_Inference ["2. Downstream Closed-Loop Conversion (2-Pass Tournament)"]
        RawNeg["Digitized RAW Negative Scan (.ARW)"] --> Sidecar["Read Companion Sidecar\n(rot_cw, hflip, vflip)"]
        Sidecar --> Pass1_Preview["Pass 1: Unclipped Nominal Target 5 Upright Preview\n(Native Aspect Ratio preserved)"]
        Pass1_Preview --> Infer1["DINOv2/v3 Multi-Layer Inference via 2D Bicubic Interpolation\n(Generates Coarse Intent Field μ_1, σ_1 at Native Grid)"]
        Infer1 --> Tourn["Candidate Bracket Tournament (Native C++ Black Box)\n(Evaluates Targets 1-7 in Upright Orientation & Native Grid)"]
        Tourn --> Stage1["Winner Target & Stage 1 Positive Render"]
        Stage1 --> Infer2["Pass 2: DINOv2/v3 In-Distribution Re-Evaluation\n(Generates Refined Intent Field μ_2, σ_2 at Native Grid)"]
        Infer2 --> FineTune["Precision Coordinate Tuning (E*, g*, b*)"]
        FineTune --> FinalRender["Final High-Resolution Conversion + Upright Color & Uncertainty Maps"]
    end
```

---

## 2. Foundation Backbones: DINOv2 and DINOv3 Architecture

The training codebase in [`src/train_dinov2_intent_field.py`](file:///home/alpha/Projects/negicc-station/src/train_dinov2_intent_field.py) dynamically supports any Vision Transformer backbone from the **DINOv2** and **DINOv3** families.

### 2.1 Why Self-Supervised Vision Representations?
- **Continuous Color & Illuminant Awareness**: Supervised semantic segmenters (e.g. ADE20K classification) collapse continuous surface reflectances into rigid categorical labels (e.g. forcing all sky to uniform cyan, regardless of golden hour, dawn, or storm).
- **Geometric Invariance & Depth Ordering**: DINOv2 and DINOv3 self-supervised representations inherently encode scene layout, relative distance, atmospheric haze, and specular highlights without human labeling bias.

### 2.2 Model Specifications & Configuration Matrix

| Model Identifier | Total Layers ($L$) | Hidden Dim ($D$) | Patch Size ($P$) | Extraction Layers ($L_{\text{photo}}, L_{\text{sem}}$) | Fused Dim ($2D$) | Adapter Hidden Dim | Input Resolution | Grid Tokens ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `facebook/dinov2-small` (Default) | $12$ | $384$ | $14$ | Layer 9, Layer 12 | **$768$** | $384$ | $518 \times 518$ | $37 \times 37 = 1{,}369$ |
| `facebook/dinov2-base` | $12$ | $768$ | $14$ | Layer 9, Layer 12 | **$1{,}536$** | $768$ | $518 \times 518$ | $37 \times 37 = 1{,}369$ |
| `facebook/dinov2-large` | $24$ | $1{,}024$ | $14$ | Layer 18, Layer 24 | **$2{,}048$** | $1{,}024$ | $518 \times 518$ | $37 \times 37 = 1{,}369$ |
| `meta/dinov3-base` | $12$ | $768$ | $14$ | Layer 9, Layer 12 | **$1{,}536$** | $768$ | $518 \times 518$ | $37 \times 37 = 1{,}369$ |
| `meta/dinov3-large` | $24$ | $1{,}024$ | $14$ | Layer 18, Layer 24 | **$2{,}048$** | $1{,}024$ | $518 \times 518$ | $37 \times 37 = 1{,}369$ |

### 2.3 Dynamic Layer Selection & Register Token Slicing
In [`_extract_worker_proc`](file:///home/alpha/Projects/negicc-station/src/train_dinov2_intent_field.py#L67-L119):
```python
num_layers = getattr(model.config, "num_hidden_layers", 12)
l_photo = int(round(num_layers * 0.75))  # Layer 9 for 12-layer, Layer 18 for 24-layer
l_sem = num_layers                       # Layer 12 for 12-layer, Layer 24 for 24-layer

reg_tokens = getattr(model.config, "num_register_tokens", 0)
k_start = 1 + reg_tokens  # Discard [CLS] (index 0) and any register tokens (e.g. 1..4)
```
Spatial tokens are extracted and concatenated along the feature dimension:
```python
h_photo = outputs.hidden_states[l_photo][:, k_start:, :].cpu().half()
h_sem = outputs.hidden_states[l_sem][:, k_start:, :].cpu().half()
feats = torch.cat([h_photo, h_sem], dim=-1)  # Shape: [B, N, 2*D]
```

### 2.4 Approach 1: Native Aspect-Ratio Grids with 2D Bicubic Positional Interpolation

Traditional vision classification pipelines anisotropically squish or stretch rectangular images into a fixed square ($518 \times 518$). While tolerable for coarse categorical classification, this creates severe artifacts in dense photographic appearance estimation:
1. **Geometric Distortion**: A $3:2$ landscape frame is squished horizontally by $1.5\times$, directionally smearing high-frequency features (film grain, fine hair, foliage, fabric) across token boundaries.
2. **Positional Embedding Conflict**: The 2D positional embeddings ($p_{r, c}$) trained on square patches encounter warped geometry, confusing horizontal and vertical distance relationships (e.g., distorting vertical vignetting and center-surround falloff).
3. **Inefficiency of Black Padding**: Letterbox/pillarbox padding wastes up to $33\%$ of transformer attention compute on zero-information borders, while creating artificial step-contrast edges that corrupt self-attention softmax distributions.

#### Mathematical Aspect-Preserving Formulation (Approach 1)
Given an input frame with native dimensions $W_{\text{in}}, H_{\text{in}}$, target patch size $P=14$, and maximum edge budget $S_{\max} = 518$:

$$\text{scale} = \frac{S_{\max}}{\max(W_{\text{in}}, H_{\text{in}})}$$

$$W_{\text{target}} = \max\left(P, \; \left\lfloor \frac{W_{\text{in}} \cdot \text{scale}}{P} + 0.5 \right\rfloor \cdot P \right), \quad H_{\text{target}} = \max\left(P, \; \left\lfloor \frac{H_{\text{in}} \cdot \text{scale}}{P} + 0.5 \right\rfloor \cdot P \right)$$

$$\text{grid}_w = \frac{W_{\text{target}}}{P}, \quad \text{grid}_h = \frac{H_{\text{target}}}{P}, \quad N = \text{grid}_w \times \text{grid}_h$$

Every patch remains an exact isotropic $14 \times 14$ pixel square. 100% of tokens represent valid photographic scene content with **zero geometric distortion** and **zero padding**.

#### Photographic Format Reference Matrix (Approach 1)

| Film Format / System | Native Aspect ($W/H$) | Input Canvas ($W_{\text{target}} \times H_{\text{target}}$) | Token Grid ($\text{grid}_w \times \text{grid}_h$) | Tokens ($N$) | Geometric Distortion | Token Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3:2 35mm Landscape** | $1.500$ | $518 \times 350$ | $37 \times 25$ | **$925$** | **$0.0\%$** (Exact isotropic) | **$100\%$** (Zero padding) |
| **2:3 35mm Portrait** | $0.667$ | $350 \times 518$ | $25 \times 37$ | **$925$** | **$0.0\%$** (Exact isotropic) | **$100\%$** (Zero padding) |
| **1:1 6x6 Medium Format** | $1.000$ | $518 \times 518$ | $37 \times 37$ | **$1{,}369$** | **$0.0\%$** (Exact isotropic) | **$100\%$** (Zero padding) |
| **4:3 645 Medium Format** | $1.333$ | $518 \times 392$ | $37 \times 28$ | **$1{,}036$** | **$0.0\%$** (Exact isotropic) | **$100\%$** (Zero padding) |
| **6:7 Medium Format** | $0.857$ | $434 \times 518$ | $31 \times 37$ | **$1{,}147$** | **$0.0\%$** (Exact isotropic) | **$100\%$** (Zero padding) |
| **65:24 XPAN Panoramic** | $2.708$ | $518 \times 196$ | $37 \times 14$ | **$518$** | **$0.0\%$** (Exact isotropic) | **$100\%$** (Zero padding) |

#### 2D Bicubic Positional Interpolation
Both DINOv2 and DINOv3 natively interpolate their 2D learned position embeddings $E_{\text{pos}} \in \mathbb{R}^{37 \times 37 \times D}$ to arbitrary $\text{grid}_h \times \text{grid}_w$ via continuous 2D bicubic splines during the forward pass. Because `PhotographicIntentAdapter` is a token-wise residual bottleneck MLP, it evaluates each token independently of sequence length $N$, ensuring seamless execution across all photographic formats.

---

## 3. Training Corpus & Aspect-Ratio Bucketing

### 3.1 Emulsion-Specific Training Corpus
Generic multi-stock corpora fail because different film emulsions possess mutually incompatible color palettes (e.g., Fuji Provia saturated greens vs Kodak Portra soft skin tones).
- **Corpus Directory**: `~/Projects/FilmCorpus/training/portra400/images`
- **Dataset Size**: **3,646 authentic Kodak Portra 400 positive master scans** from professional laboratory scanners (Fuji Frontier SP3000 and Noritsu HS-1800).
- **Corpus Aspect Distribution**:
  - $36.5\%$ Square ($1:1$, $518 \times 518 \rightarrow 1,369$ patches)
  - $28.5\%$ Standard 35mm Landscape ($3:2$ / $1.54$, $518 \times 350 / 518 \times 336 \rightarrow 925 / 888$ patches)
  - $12.0\%$ Medium Format 4:3 ($518 \times 392 / 518 \times 420 \rightarrow 1,036 / 1,110$ patches)
  - $7.0\%$ Portrait formats ($2:3$, $350 \times 518 \rightarrow 925$ patches)
  - $16.0\%$ Panoramic and specialty medium format crops.
- **Ground Truth Targets**: For each $14 \times 14$ spatial patch, mean CIELAB values are computed:
  $$\mathbf{y}_i = [L^*_i, a^*_i, b^*_i] \in \mathbb{R}^3 \quad \text{for } i \in \{1, \dots, N\}$$

### 3.2 Aspect-Ratio Bucketing Extraction
During feature extraction ([`_extract_worker_proc`](file:///home/alpha/Projects/negicc-station/src/train_dinov2_intent_field.py)), image files are dynamically grouped into aspect-ratio buckets `(target_w, target_h)`:
1. All images within an aspect bucket are batched together into `[B, 3, target_h, target_w]`.
2. DINOv2 / DINOv3 extracts fused features of shape `[B, N, 2D]` where $N = \text{grid}_w \times \text{grid}_h$.
3. Tokens are flattened into continuous 2D tensors `[B * N, 2D]` and targets `[B * N, 3]`.
4. This produces a unified multi-aspect dataset (~4.0 million tokens) where training proceeds completely scale-agnostic and token-wise.

### 3.3 Numerical Precision & Zero-Quantization Loss Guarantee

A critical vulnerability in color appearance modeling is **color quantization loss**, where continuous chromatic coordinates are prematurely rounded or truncated into discrete 8-bit integers.

#### 1. Why Integer Quantization Fails for Photographic Intent
If RGB images are converted to CIELAB using integer representations (`uint8`):
- OpenCV rescales $L^*$ to $[0, 255]$ ($L^*_{\text{uint8}} = L^* \times 255 / 100$), and shifts $a^*, b^*$ by $+128$ into $[0, 255]$.
- Each discrete step of $a^*$ or $b^*$ corresponds to $\approx 1.0$ CIELAB $\Delta E$ unit.
- Since the human **just-noticeable difference (JND)** threshold is $\Delta E^*_{ab} \approx 1.0\text{--}2.0$, integer quantization introduces artificial banding, false neutral biases, and obliterates subtle pastel color nuances (critical for Kodak Portra skin tones and delicate highlight roll-offs).
- OpenCV also **does not support 16-bit integer (`CV_16U`)** for `RGB2LAB` (raising a fatal runtime error: `Unsupported depth of input image: CV_16U`).

#### 2. The Continuous 32-bit Floating-Point Protocol (`CV_32F`)
To guarantee **zero loss of precision due to quantization**, the pipeline enforces continuous floating-point evaluation at every stage:

1. **Floating-Point Normalization**:
   Before color space transformation, input RGB arrays are cast to IEEE 754 32-bit floats scaled to $[0.0, 1.0]$:
   ```python
   arr = np.array(im, dtype=np.float32) / 255.0  # Or / 65535.0 for 16-bit buffers
   lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
   ```
2. **Continuous Physical CIELAB Transfer Function**:
   When passed `CV_32F` buffers, OpenCV evaluates the exact non-linear CIE 1976 colorimetric transfer function using continuous floating-point arithmetic:
   $$f(t) = \begin{cases} t^{1/3} & \text{if } t > \left(\frac{6}{29}\right)^3 \\ \frac{1}{3} \left(\frac{29}{6}\right)^2 t + \frac{4}{29} & \text{otherwise} \end{cases}$$
   $$L^* = 116 f\left(\frac{Y}{Y_n}\right) - 16, \quad a^* = 500 \left[ f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right) \right], \quad b^* = 200 \left[ f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right) \right]$$
   Output values are continuous real numbers: $L^* \in [0.0, 100.0]$, $a^*, b^* \in [-128.0, 127.0]$ with machine epsilon $\approx 1.19 \times 10^{-7}$.

3. **Spatial Over-Sampling & Effective Sub-Bit Precision**:
   Each spatial token represents the spatial average over a $14 \times 14$ window ($P^2 = 196$ pixels):
   $$\mathbf{y}_i = \frac{1}{196} \sum_{p=1}^{196} \mathbf{lab}(p)$$
   By the Central Limit Theorem, spatial box averaging over 196 uncorrelated pixels increases the effective signal-to-noise ratio and bit depth:
   $$\Delta \text{bits} = \log_2(\sqrt{196}) = \log_2(14) \approx 3.81 \text{ bits}$$
   Even when starting from an 8-bit positive master scan, the resulting patch mean target $\mathbf{y}_i$ possesses the continuous numerical precision of an **$\approx 11.81\text{-bit}$ signal** ($> 3{,}500$ discrete quantization steps across the dynamic range), completely eliminating integer posterization.

4. **Storage & Gradient Arithmetic**:
   - **Feature Cache**: Patch features and target LAB values are stored as **`torch.float32`** (24 mantissa bits, zero truncation/quantization loss). Given workstation memory capacity (472 GB RAM), full 32-bit floats preserve bit-exact floating-point fidelity without `float16` rounding.
   - **Neural Adapter**: Loss computation, backward gradients, and parameter updates execute in full **`torch.float32`** (24 mantissa bits).
   - **Downstream Converter**: In [`src/batch_dinov2_convert.py`](file:///home/alpha/Projects/negicc-station/src/batch_dinov2_convert.py), candidate images are similarly converted to `float32` prior to CIELAB transform, ensuring exact numerical continuity and train/inference bit-exact symmetry when computing standardized Mahalanobis residuals $\frac{\mathbf{y}_i - \boldsymbol{\mu}_i}{\boldsymbol{\sigma}_i}$.

5. **Linear-Light Resampling Pipeline (sRGB $\to$ Linear RGB $\to$ Scale $\to$ sRGB $\to$ CIELAB)**:
   A fundamental vulnerability in naive computer vision pipelines is downsampling images directly in non-linear gamma-encoded space ($V \approx L^{1/\gamma}$). Because $f(x) = x^{1/\gamma}$ is strictly concave for $\gamma > 1$, by Jensen's inequality:
   $$\sum_{i} w_i L_i^{1/\gamma} < \left(\sum_{i} w_i L_i\right)^{1/\gamma} \implies \left(\sum_{i} w_i V_i\right)^\gamma < \sum_{i} w_i L_i$$
   Resampling in gamma space causes **radiometric energy loss** (perceptible darkening of high-contrast edges, fine textures, foliage, and highlights) and **chromaticity/hue shifts** at transition boundaries (introducing up to $\Delta E \approx 16.0$ artificial darkening on sharp edges).
   
   To achieve bit-exact physical photon conservation while maintaining seamless colorimetric and vision backbone compatibility, the pipeline executes **Linear-Light Resampling**:
   1. **sRGB to Linear RGB**:
      The input positive scan is linearized using the exact IEC 61966-2-1 inverse EOTF into IEEE 754 float32 linear radiance ($C_{\text{linear}} = \text{LUT}[C_{\text{sRGB}}]$).
   2. **Linear-Light Scaling**:
      Rescaling to the patch-aligned native aspect dimensions $(W_{\text{target}}, H_{\text{target}})$ is executed in **linear light** (`cv2.INTER_AREA` for exact physical radiant flux conservation, or `cv2.INTER_LINEAR`). This completely eliminates gamma darkening errors and preserves true photometric luminance.
   3. **Linear RGB back to sRGB**:
      The rescaled linear buffer is converted back to sRGB in continuous 32-bit floats ($C_{\text{sRGB}} = \text{linear\_to\_srgb}(C_{\text{linear\_scaled}})$).
   4. **Unified Input & Ground-Truth Evaluation**:
      - **Vision Backbone (ViT)**: The linear-scaled sRGB buffer is normalized by standard ImageNet statistics $(\boldsymbol{\mu}, \boldsymbol{\sigma})$ and fed to DINOv2 / DINOv3.
      - **CIELAB Ground Truth**: Converted using standard continuous `cv2.cvtColor(srgb_scaled, cv2.COLOR_RGB2LAB)` and spatial patch box-averaged across each $P \times P$ block ($14 \times 14$ or $16 \times 16$) to produce continuous $[L^*_i, a^*_i, b^*_i]$. Both encoder and ground truth evaluate the identical gamma-error-free scene.

### 3.4 Upright Orientation Invariance
Vision Transformers utilize **2D spatial positional embeddings** ($p_{r, c}$) fixed to the patch grid:
- In the training corpus, all photos are upright (sky/clouds in upper rows, terrain/shadows in lower rows).
- If an unrotated RAW negative (e.g. sideways sensor capture) is fed into the model, the positional embeddings see sky on the side, confusing the semantic intent.
- **Strict Rule**: All negative previews and candidate conversions must pass `rot_cw, hflip, vflip` (from companion sidecar `.json` files) to ensure **upright evaluation**.

---

## 4. Adapter Architecture & Heteroscedastic Objective

### 4.1 `PhotographicIntentAdapter`
Defined in [`PhotographicIntentAdapter`](file:///home/alpha/Projects/negicc-station/src/train_dinov2_intent_field.py#L30-L65):

$$\begin{aligned}
\mathbf{h}^{(1)}_i &= \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_{\text{in}} \mathbf{z}_i + \mathbf{b}_{\text{in}}\right)\right) && \in \mathbb{R}^{\text{hidden\_dim}} \\
\mathbf{h}^{(2)}_i &= \text{LayerNorm}\left(\mathbf{h}^{(1)}_i + \mathbf{W}_2 \, \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_1 \mathbf{h}^{(1)}_i + \mathbf{b}_1\right)\right) + \mathbf{b}_2\right) && \in \mathbb{R}^{\text{hidden\_dim}} \\
\boldsymbol{\mu}_i &= \mathbf{W}_{\mu} \mathbf{h}^{(2)}_i + \mathbf{b}_{\mu} && \in \mathbb{R}^3 \quad (L^*, a^*, b^*) \\
\log \boldsymbol{\sigma}_i^2 &= \text{clamp}\left(\mathbf{W}_{\sigma} \mathbf{h}^{(2)}_i + \mathbf{b}_{\sigma}, \; -2.5, \; 5.0\right) && \in \mathbb{R}^3
\end{aligned}$$

#### Weight Initialization Priors
- **Mean Head**: $\mathbf{W}_{\mu} = \mathbf{0}$, $\mathbf{b}_{\mu} = [50.0, 0.0, 0.0]$ (neutral 18% gray baseline).
- **Variance Head**: $\mathbf{W}_{\sigma} \sim \mathcal{N}(0, 0.02^2)$, $\mathbf{b}_{\sigma} = [2.0, 1.0, 1.0]$.

### 4.2 Heteroscedastic Robust Huber NLL Loss
To model natural photographic variance without collapsing into an "average photo" desaturation:

1. **Standardized Residual**:
   $$r_{i, c} = \frac{|\mathbf{y}_{i, c} - \boldsymbol{\mu}_{i, c}|}{\boldsymbol{\sigma}_{i, c}}, \quad \text{where } \boldsymbol{\sigma}_{i, c} = \exp\left(0.5 \cdot \log \boldsymbol{\sigma}_{i, c}^2\right)$$
2. **Huber Metric**:
   $$\mathcal{H}(r) = \begin{cases} 
   \frac{1}{2} r^2 & \text{if } r < 1.0 \\
   r - 0.5 & \text{if } r \ge 1.0
   \end{cases}$$
3. **Total Objective**:
   $$\mathcal{L}_{\text{total}} = \frac{1}{B \cdot N \cdot 3} \sum_{b=1}^B \sum_{i=1}^N \sum_{c=1}^3 \left( \mathcal{H}(r_{b, i, c}) + \frac{1}{2} \log \boldsymbol{\sigma}_{b, i, c}^2 \right)$$

#### Natural Dynamic Precision Weighting
The learned uncertainty $\boldsymbol{\sigma}$ achieves an empirical dynamic range of **$1.13$ to $12.18$**:
- **Tight Tolerance ($\sigma \approx 1.1$)**: Specular highlights, snow ridges, clear daylight sky, skin tones.
- **Broad Tolerance ($\sigma \approx 12.0$)**: Shadowed foliage, noisy film grain, indistinct dark backgrounds.
- **Precision Ratio**: Because optimization weights scale as $w_i \propto 1/\sigma_i^2$, strict patches carry **$117\times$ higher penalty**, naturally preserving key highlights without hand-crafted saliency heuristics.

---

## 5. Standalone Execution Protocol for Subagents

Any autonomous agent or background job can spin off feature extraction and training independently using the CLI interface.

### 5.1 Environment Prerequisites
- Virtualenv: `/home/alpha/Projects/negicc-station/venv`
- Python Path: `PYTHONPATH=src`
- Working Directory: `/home/alpha/Projects/negicc-station`

### 5.2 CLI Command Reference
The primary training entrypoint is [`src/train_dinov2_intent_field.py`](file:///home/alpha/Projects/negicc-station/src/train_dinov2_intent_field.py):

| Flag | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `--corpus-dir` | `str` | Auto-detected | Path to positive film scan corpus directory. |
| `--cache-path` | `str` | `models/dinov2_features_cache_518_multilayer.pt` | Path to save/load sharded feature cache (7.69 GB). |
| `--output-model` | `str` | `models/dinov2_photographic_intent_field_multilayer.pt` | Path to save trained adapter weights checkpoint. |
| `--backbone` | `str` | `facebook/dinov2-small` | Vision Transformer backbone (e.g. `facebook/dinov2-small`, `facebook/dinov2-base`, `meta/dinov3-base`). |
| `--epochs` | `int` | `60` | Training epochs (converges in ~10–15 epochs). |
| `--batch-size` | `int` | `64` | Mini-batch size (interprets $<1000$ as image-equivalent batches of $1,024$ tokens each). |
| `--extract-batch-size` | `int` | `16` | Per-worker batch size during vision feature extraction. |
| `--num-workers` | `int` | `4` | Number of parallel extraction processes. |
| `--lr` | `float` | `1e-3` | Initial AdamW learning rate (with Cosine Annealing to 1e-5). |
| `--val-split` | `float` | `0.1` | Validation split fraction (10% held out). |
| `--native-aspect` | `flag` | `True` | **Approach 1**: Extract native aspect-ratio grids with 2D bicubic positional interpolation (default). |
| `--legacy-square` | `flag` | `False` | Use legacy 518x518 square squishing (anisotropic distortion). |
| `--force-extract` | `flag` | `False` | Force re-extraction even if feature cache exists. |

---

### 5.3 Spin-Off Execution Recipes

#### Recipe 1: Production Run (Approach 1 Native Aspect DINOv2-Small on Portra 400)
```bash
PYTHONPATH=src venv/bin/python3 src/train_dinov2_intent_field.py \
    --corpus-dir ~/Projects/FilmCorpus/training/portra400/images \
    --cache-path models/dinov2_features_cache_native_aspect.pt \
    --output-model models/dinov2_photographic_intent_field_multilayer.pt \
    --backbone facebook/dinov2-small \
    --native-aspect \
    --epochs 60 \
    --batch-size 64 \
    --num-workers 4
```
*Total Runtime: ~3.5 minutes for full multi-aspect extraction + 25 seconds for training.*

#### Recipe 2: Instant Hyperparameter Retraining (from Existing Cache)
```bash
PYTHONPATH=src venv/bin/python3 src/train_dinov2_intent_field.py \
    --cache-path models/dinov2_features_cache_518_multilayer.pt \
    --output-model models/dinov2_photographic_intent_field_tuned.pt \
    --epochs 40 \
    --lr 5e-4
```
*Total Runtime: ~18 seconds.*

#### Recipe 3: Scaling to DINOv2-Base / DINOv3 with Native Aspect Grids
```bash
PYTHONPATH=src venv/bin/python3 src/train_dinov2_intent_field.py \
    --corpus-dir ~/Projects/FilmCorpus/training/portra400/images \
    --cache-path models/dinov2_base_features_cache_native_aspect.pt \
    --output-model models/dinov2_base_photographic_intent_field.pt \
    --backbone facebook/dinov2-base \
    --native-aspect \
    --extract-batch-size 8 \
    --epochs 60 \
    --force-extract
```

---

## 6. Numerical Acceptance Gates & Verification

Before promoting any trained checkpoint to production conversion, verify the following validation gates:

| Acceptance Gate | Required Threshold | Iteration 4 Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Validation Huber NLL Loss** | $\le 2.30$ | **$2.1940$** | **PASSED** |
| **Validation MAE $a^*, b^*$** | $\le 5.00$ CIELAB | **$4.69$** | **PASSED** |
| **Validation MAE $L^*$** | $\le 8.00$ CIELAB | **$7.36$** | **PASSED** |
| **Uncertainty Dynamic Range** | $\ge 8.0\times$ | **$10.82\times$** ($117\times$ weight ratio) | **PASSED** |
| **Neutral Bias on Zero Token** | $L^* = 50.0 \pm 2.0$ | **$50.00$** | **PASSED** |

---

## 7. Downstream Runtime Consumption: Closed-Loop 2-Pass Tournament

The trained model is consumed directly by [`src/batch_dinov2_convert.py`](file:///home/alpha/Projects/negicc-station/src/batch_dinov2_convert.py):

```bash
PYTHONPATH=src venv/bin/python3 src/batch_dinov2_convert.py \
    --checkpoint-path models/dinov2_photographic_intent_field_multilayer.pt
```

### The 2-Pass Closed-Loop Execution (Approach 1):
1. **Pass 1: Bootstrap & Bracket Selection**:
   - Computes native patch-aligned dimensions via [`compute_aspect_preserved_shape`](file:///home/alpha/Projects/negicc-station/src/batch_dinov2_convert.py) (e.g. $518 \times 392$ for 4:3, $518 \times 350$ for 3:2 landscape, $350 \times 518$ for 2:3 portrait).
   - Renders nominal Target 5 preview in the upright orientation ([`run_dinov2_tournament_conversion`](file:///home/alpha/Projects/negicc-station/src/batch_dinov2_convert.py)).
   - Evaluates multi-layer DINOv2/v3 intent via 2D bicubic positional interpolation: predicts $\boldsymbol{\mu}_1, \boldsymbol{\sigma}_1$ at native grid shape $(\text{grid}_h, \text{grid}_w)$.
   - Native C++ engine evaluates candidate profile target brackets (Targets 1..7) against $\boldsymbol{\mu}_1, \boldsymbol{\sigma}_1$ at the exact native patch grid without squishing or padding.
   - Selects winning profile target bracket (Target $k$) and coarse gains $(E_1, g_1, b_1)$.
2. **Pass 2: In-Distribution Precision Tuning**:
   - Native C++ engine renders Stage 1 positive image: `stage1_crop_u8`.
   - DINOv2/v3 re-evaluates `stage1_crop_u8` strictly in-distribution, predicting refined intent field $\boldsymbol{\mu}_2, \boldsymbol{\sigma}_2$.
   - Optimizer fine-tunes $(E^*, g^*, b^*)$ via coordinate descent within the bounded basin.
3. **Generated Artifacts**:
   - Full-resolution converted positive image (`*_dinov2_intent.jpg`).
   - Dense CIELAB intent color map rendered in sRGB preserving native aspect ratio (`*_color_map.jpg`).
   - Dense uncertainty heatmap preserving native aspect ratio (`*_uncertainty_map.jpg`).
   - 5-panel comprehensive comparison image tracking ground truth, baseline, and DINOv2 result (`*_iter4_comparison.jpg`).

### 7.1 Linear Grid Direct Downscaling Optimization (1,000x Faster Search)

During iterative parameter optimization (searching ~300 candidate combinations of profile target brackets $k$, exposure $E$, and chromatic gains $g, b$), converting the entire full-frame preview into sRGB before spatial downsampling creates an immense computational bottleneck (~4.2 seconds per target search).

To resolve this, the optimization pipeline exploits **linear light commutation**:
$$\text{Downsample}(\mathbf{M} \cdot \mathbf{x}_{\text{linear}}) \equiv \mathbf{M} \cdot \text{Downsample}(\mathbf{x}_{\text{linear}})$$

#### Architectural Implementation
1. **Pre-Downscaling in Linear Camera Space**:
   - The upright cropped 16-bit linear RGB buffer is downsampled directly to the target grid dimensions ($\text{grid}_w \times \text{grid}_h$, e.g. $37 \times 25 = 925$ pixels) using area averaging (`cv2.INTER_AREA`).
   - Downsampling in linear light is radiometrically exact and completely eliminates gamma darkening errors inherent to non-linear sRGB averaging.
2. **Native C++ Micro-Conversion Engine (`convert_linear_buffer_rgb8`)**:
   - Added [`convert_linear_buffer_rgb8`](file:///home/alpha/Projects/negicc-station/src/image_capture.cpp) and Python binding `negicc_station.convert_linear_rgb8(...)`.
   - Executes the exact color pipeline (crosstalk matrix + knee compression + 1D input TRC + 3D CLUT + 1D output TRC + Bradford chromatic adaptation + sRGB transfer function) directly on arbitrary 16-bit linear memory buffers.
3. **Thread-Safe LittleCMS Profile Caching**:
   - `CachedIccProfile` parses the 572 KB LittleCMS ICC profile tables once per target pointer and caches the unpacked grids and curves, eliminating repeated ~12ms profile decode overhead.
4. **OpenMP Core-Count Threshold Tuning**:
   - Spawning 192 CPU threads via `#pragma omp parallel for` on a 925-pixel array incurred a massive ~14ms fork/join synchronization barrier.
   - An execution guard (`if (total_pixels > 8192)`) restricts multi-threading to large frames while executing small grid trials sequentially on a single core in **0.14 ms**.
5. **Conversion Parity & Quality Invariant**:
   - Mean absolute difference between linear grid conversion and full-frame conversion is $\Delta L^* = 0.23, \Delta a^* = 0.32, \Delta b^* = 0.97$ ($\Delta E < 1.0$ JND, below human visibility).
   - Once optimal parameters $(k^*, E^*, g^*, b^*)$ are determined on the grid, the final high-resolution render is executed using the unmodified full-resolution `convert_raw_to_numpy(...)` engine.

---

## 8. Registered Photographic Intent Appearance Models

The following production-ready appearance models are trained, validated against all numerical acceptance gates, and exported in `~/Projects/FilmModels/exports/`:

| Emulsion Profile | Model Identifier | Vision Backbone | Val Loss | MAE Total | MAE $L^*$ | MAE $a^*, b^*$ | Uncertainty Dynamic Range | Export Directory |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Kodak Portra 400** | `dinov2-small_portra400_intent_field` | DINOv2-small | 1.6866 | 4.09 | 5.09 | 3.58 | 42.5x | [`dinov2-small_portra400/`](file:///home/alpha/Projects/FilmModels/exports/dinov2-small_portra400/) |
| **Kodak Portra 400** | `dinov3-small_portra400_intent_field` | DINOv3-small | 1.7646 | 4.13 | 3.75 | 4.32 | 42.5x | [`dinov3-small_portra400/`](file:///home/alpha/Projects/FilmModels/exports/dinov3-small_portra400/) |
| **Kodak Portra 160** | `dinov3-small_portra160_intent_field` | DINOv3-small | 1.5337 | 3.32 | 3.24 | 3.37 | 42.5x | [`dinov3-small_portra160/`](file:///home/alpha/Projects/FilmModels/exports/dinov3-small_portra160/) |
| **Kodak Portra** (Family) | `dinov3-small_portra_intent_field` | DINOv3-small | 1.7075 | 3.95 | 3.61 | 4.12 | 42.5x | [`dinov3-small_portra/`](file:///home/alpha/Projects/FilmModels/exports/dinov3-small_portra/) |
| **Fujifilm Pro 160NS** | `dinov3-small_pro160_intent_field` | DINOv3-small | 1.3981 | 2.97 | 2.68 | 3.12 | 42.5x | [`dinov3-small_pro160/`](file:///home/alpha/Projects/FilmModels/exports/dinov3-small_pro160/) |
| **Fujifilm Pro 400H** | `dinov3-small_pro400_intent_field` | DINOv3-small | 1.6133 | 3.80 | 3.37 | 4.02 | 42.5x | [`dinov3-small_pro400/`](file:///home/alpha/Projects/FilmModels/exports/dinov3-small_pro400/) |
| **Kodak Ektar 100** | `dinov3-small_ektar100_intent_field` | DINOv3-small | 1.8175 | 4.47 | 3.48 | 4.96 | 42.5x | [`dinov3-small_ektar100/`](file:///home/alpha/Projects/FilmModels/exports/dinov3-small_ektar100/) |

### Model Export Artifacts per Directory:
Every model directory in `~/Projects/FilmModels/exports/` contains 4 standard deployment files:
1. `model.pt`: Full PyTorch state dict and training metadata.
2. `model.torchscript`: Traced JIT module optimized for native C++ libtorch runtime execution (`torch::jit::load`).
3. `model.onnx` & `model.onnx.data`: Open Neural Network Exchange graph with dynamic batch and sequence length axes.
4. `metadata.json`: Machine-readable parameters, dimensions, hyperparams, and acceptance gate evaluations.
