# Appearance Model Training Pipeline (DINOv3-Small)

This directory contains the complete source code, build rules, and specifications for training **Photographic Intent Field Appearance Models** from a vision foundation backbone (`Tooony133/dinov3-vits16-pretrain-lvd1689m`) and a corpus of authentic film scans.

---

## 1. Overview & Mathematical Architecture

An appearance model predicts the locally plausible spatial color distribution $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i)$ in CIE $L^*a^*b^*$ space for a given film stock. The model acts as the photographic intent target for downstream exposure ($E$) and chromatic gain ($g, b$) optimization during negative conversion.

### The Two-Paths Extraction Pipeline
1. **Path A (Vision Backbone)**:
   - Resizes input scan to patch-aligned dimensions preserving native aspect ratio (e.g. $512 \times 352$ for 3:2 landscape, $352 \times 512$ for 2:3 portrait).
   - Feeds image through frozen DINOv3-small ($P=16\text{ px}$).
   - Extracts fused tokens from Layer 9 (photographic features) and Layer 12 (semantic features) into a 768-D representation.
   - Discards token indices $0..4$ (1 `[CLS]` + 4 register tokens) to retain only spatial patch tokens.
2. **Path B (Ground Truth Radiometric Target)**:
   - Performs exact IEC 61966-2-1 inverse sRGB EOTF conversion to physical linear energy.
   - Downsamples in linear optical space via box area-averaging (`cv2.INTER_AREA`).
   - Converts to float32 CIE $L^*a^*b^*$.

### The PhotographicIntentAdapter
A lightweight residual MLP mapping 768-D tokens to 3-D CIELAB:
$$\mathbf{h}^{(1)} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_{\text{in}} \mathbf{z} + \mathbf{b}_{\text{in}}))$$
$$\mathbf{h}^{(2)} = \text{LayerNorm}(\mathbf{h}^{(1)} + \mathbf{W}_2 \text{GELU}(\text{LayerNorm}(\mathbf{W}_1 \mathbf{h}^{(1)} + \mathbf{b}_1)) + \mathbf{b}_2)$$
$$\boldsymbol{\mu} = \mathbf{W}_\mu \mathbf{h}^{(2)} + \mathbf{b}_\mu, \quad \log \boldsymbol{\sigma}^2 = \text{clamp}(\mathbf{W}_\sigma \mathbf{h}^{(2)} + \mathbf{b}_\sigma, -2.5, 5.0)$$

- **Physical Prior Initialization**:
  - Mean head: $\mathbf{b}_\mu = [50.0, 0.0, 0.0]$ (neutral photographic mid-gray).
  - Variance head: $\mathbf{b}_\sigma = [2.0, 1.0, 1.0]$ (sensible initial tolerance).

### Heteroscedastic Robust Huber Loss
Standardized residual:
$$r_{i, c} = \frac{|\mathbf{y}_{i, c} - \boldsymbol{\mu}_{i, c}|}{\boldsymbol{\sigma}_{i, c}}$$
$$\mathcal{H}(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } r < 1.0 \\ r - 0.5 & \text{if } r \ge 1.0 \end{cases}$$
$$\mathcal{L} = \frac{1}{B \cdot N \cdot 3} \sum_{b, i, c} \left( \mathcal{H}(r_{b, i, c}) + \frac{1}{2} \log \boldsymbol{\sigma}_{b, i, c}^2 \right)$$

---

## 2. Quickstart: Building and Training Models

### Using the Makefile
```bash
# 1. Download backbone and train Portra 400 appearance model using metadata.json:
make all

# 2. Run a fast test verification (2 epochs, 10 images):
make test

# 3. Explicitly train Portra 400:
make train_portra400 CORPUS_DIR=/path/to/FilmCorpus/training EPOCHS=60

# 4. Clean temporary build caches:
make clean
```

### Using Python Directly
```bash
python3 train_appearance_model.py \
    --metadata ../models/dinov3-small_portra400/metadata.json \
    --corpus-dir /home/alpha/Projects/FilmCorpus/training \
    --output-dir ../models/dinov3-small_portra400 \
    --epochs 60 \
    --batch-size 64 \
    --max-samples 400 \
    --device cpu
```

---

## 3. Specification of `metadata.json`

Every exported appearance model includes a `metadata.json` document that fully specifies the model configuration, provenance, arithmetic guarantees, and validation benchmarks.

### Example `metadata.json` Schema
```json
{
  "model_name": "dinov3-small_portra400_intent_field",
  "profile": "portra400",
  "backbone": "dinov3-small",
  "aspect_mode": "approach_1_native_aspect_bicubic",
  "is_native_aspect": true,
  "arithmetic_precision": "float32",
  "mantissa_bits": 24,
  "resampling_method": "srgb_to_linear_rgb_scale_to_srgb_to_cielab",
  "color_pipeline": "Exact IEC 61966-2-1 inverse EOTF -> cv2.INTER_AREA linear scale -> forward EOTF -> cv2.COLOR_RGB2LAB (float32)",
  "backbone_details": {
    "model_id": "Tooony133/dinov3-vits16-pretrain-lvd1689m",
    "image_size": 512,
    "patch_size": 16,
    "grid_size": 32,
    "num_patches": 1024,
    "raw_dim": 384,
    "fused_dim": 768,
    "adapter_hidden_dim": 384,
    "token_start_idx": 5,
    "description": "Meta DINOv3 Small (ViT-S/16, 512x512, 1,024 patches, Layers 9+12 fused 768-D)"
  },
  "parameters": 595590,
  "in_dim": 768,
  "hidden_dim": 384,
  "export_date": "2026-09-12T02:05:31.388755+00:00",
  "input_dimensions": {
    "batch_size": "variable",
    "num_patches": "variable_native_aspect (e.g. 992 for 3:2, 768 for 4:3, 1024 for 1:1)",
    "token_dim": 768
  },
  "output_dimensions": {
    "mu": ["batch_size", "num_patches", 3],
    "sigma": ["batch_size", "num_patches", 3],
    "log_var": ["batch_size", "num_patches", 3]
  },
  "color_space": "CIE L*a*b*",
  "channel_ranges": {
    "L*": [0.0, 100.0],
    "a*": [-128.0, 127.0],
    "b*": [-128.0, 127.0]
  },
  "metrics": {
    "epoch": 60,
    "val_loss": 1.7646,
    "val_mae": 4.1305,
    "val_mae_L": 3.7477,
    "val_mae_ab": 4.3219,
    "train_loss": 1.7574,
    "train_mae": 4.1021
  },
  "acceptance_gates": {
    "gates": {
      "gate_val_loss_le_2_30": true,
      "gate_mae_ab_le_5_00": true,
      "gate_mae_L_le_8_00": true
    },
    "all_passed": true
  },
  "exported_files": {
    "pytorch_weights": "model.pt",
    "torchscript": "model.torchscript",
    "onnx": "model.onnx",
    "metadata": "metadata.json"
  }
}
```

### Key Fields Explained
- `profile`: The canonical film emulsion name (e.g. `portra400`, `gold200`, `ektar100`, `pro400h`).
- `backbone`: Architecture identifier (`dinov3-small`).
- `token_start_idx`: Index where spatial patch tokens start. For DINOv3, this is `5` because index `0` is the `[CLS]` token and indices `1..4` are Meta DINOv3 register tokens.
- `in_dim`: Concatenated token dimension across Layer 9 ($384\text{-D}$) and Layer 12 ($384\text{-D}$) = $768\text{-D}$.
- `resampling_method`: Guarantee that ground truth patches were computed in linear optical energy before CIELAB mapping.
- `acceptance_gates`: Numerical quality standards that must be met before deploying the model:
  - $\text{Val Loss} \le 2.30$
  - $\text{MAE } ab^* \le 5.00$
  - $\text{MAE } L^* \le 8.00$
