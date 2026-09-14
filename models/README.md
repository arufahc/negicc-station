# Photographic Intent Appearance Models

This directory houses the **Photographic Intent Field Appearance Models** used by `negicc-station` for neural film negative inversion.

---

## 1. Architecture Overview

An appearance model is composed of two components:
1. **Vision Foundation Backbone (`dinov3-small`)**:
   - Meta DINOv3 Small (`Tooony133/dinov3-vits16-pretrain-lvd1689m`), a Vision Transformer with $16 \times 16\text{ px}$ patches and 4 register tokens.
   - Frozen during training and inference.
   - Pulled automatically to `models/dinov3-small/` during the build (`make models`), and excluded from Git version control.
2. **Photographic Intent Adapter (`PhotographicIntentAdapter`)**:
   - A lightweight residual bottleneck MLP (~595K parameters, ~2.3 MB) trained specifically on positive scans of authentic film emulsions.
   - Fuses multi-layer representations: Layer 9 (fine-grained photographic/dye characteristics) + Layer 12 (high-level semantic context) into a 768-D token field.
   - Predicts per-patch CIE $L^*a^*b^*$ color distributions $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i)$.
   - Checked directly into Git under `models/<backbone>_<profile>/` (e.g. `models/dinov3-small_portra400/`).

---

## 2. Directory Layout

```
models/
├── dinov3-small/                     # Base HuggingFace backbone weights (downloaded via `make models`, gitignored)
│   ├── config.json
│   └── model.safetensors
├── dinov3-small_portra400/           # Checked-in Kodak Portra 400 appearance model
│   ├── metadata.json                 # Architecture, training parameters, and benchmark metrics
│   ├── model.pt                      # PyTorch adapter weights checkpoint
│   ├── model.torchscript             # Traced TorchScript module for C++ runtime
│   ├── model.onnx                    # Open Neural Network Exchange format
│   ├── model.onnx.data               # Serialized ONNX tensor data
│   └── README.md
├── portra400 -> dinov3-small_portra400 # Convenience alias symlink
└── README.md                         # This guide
```

---

## 3. How the Runtime Loads Models

The runtime factory in `src/batch_dinov2_convert.py` (and `src/dinov3_convert.py`) loads models via:

```python
from batch_dinov2_convert import load_intent_model

# Loads by path or profile name
model = load_intent_model("models/dinov3-small_portra400")
```

1. **Backbone Resolution**: The loader checks `models/dinov3-small/` locally first. If the local directory exists, it loads directly from disk without needing network access or contacting HuggingFace Hub.
2. **Adapter Initialization**: The loader reads `metadata.json` to configure token dimensions (`in_dim=768`, `hidden_dim=384`, `token_start_idx=5`), loads `model.pt`, and attaches the adapter to the backbone.

---

## 4. How to Build Your Own Appearance Model

Follow these steps to train and deploy an appearance model for a new film emulsion (e.g. Kodak Gold 200, Fuji Pro 400H, Kodak Tri-X 400, Kodak Ektar 100).

### Step 1: Collect a Film Positive Scan Corpus
Acquire authentic positive scans of the target film stock (ideally Frontier SP3000 or Noritsu HS-1800 lab scans).
- Recommended dataset size: 100 to 500 scans covering varied subjects (portraits, landscapes, architecture, golden hour, overcast, flash).
- Store them in a folder: e.g. `/home/alpha/Projects/FilmCorpus/training/gold200/images/`.

### Step 2: Create or Point to a Training Configuration
You can copy an existing `metadata.json` as a baseline template:

```bash
mkdir -p models/dinov3-small_gold200
cp models/dinov3-small_portra400/metadata.json models/dinov3-small_gold200/metadata.json
```

Edit `models/dinov3-small_gold200/metadata.json` to change the profile name:
```json
{
  "model_name": "dinov3-small_gold200_intent_field",
  "profile": "gold200",
  "backbone": "dinov3-small"
}
```

### Step 3: Run the Training Pipeline
Navigate to `training/` and run `train_appearance_model.py` (or use `make`):

```bash
cd training

# Using train_appearance_model.py
python3 train_appearance_model.py \
    --metadata ../models/dinov3-small_gold200/metadata.json \
    --profile gold200 \
    --corpus-dir /path/to/FilmCorpus/training \
    --output-dir ../models/dinov3-small_gold200 \
    --epochs 60 \
    --batch-size 64 \
    --max-samples 400 \
    --device cpu
```

The script will:
1. Load or download the local `dinov3-small` backbone.
2. Group corpus images by aspect ratio (3:2, 4:3, 1:1, 2:3).
3. Extract multi-layer spatial patch tokens (Layers 9 + 12) with linear-light radiometric area resampling.
4. Train the residual adapter using Heteroscedastic Huber loss on standardized residuals.
5. Export `model.pt`, `model.torchscript`, `model.onnx`, and updated `metadata.json`.

### Step 4: Verify Acceptance Gates
Inspect `models/dinov3-small_gold200/metadata.json` and check `acceptance_gates`:
```json
"acceptance_gates": {
  "gates": {
    "gate_val_loss_le_2_30": true,
    "gate_mae_ab_le_5_00": true,
    "gate_mae_L_le_8_00": true
  },
  "all_passed": true
}
```
If `all_passed` is `true`, your model is ready for production negative conversion.

### Step 5: Create Symlink and Test
```bash
cd models
ln -s dinov3-small_gold200 gold200

# Test negative conversion using your new model:
python3 src/dinov3_convert.py --model-path models/dinov3-small_gold200
```

---

## 5. Description of `metadata.json` Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `model_name` | `string` | Unique identifier (e.g. `dinov3-small_portra400_intent_field`). |
| `profile` | `string` | Canonical film stock name (`portra400`, `gold200`, etc.). |
| `backbone` | `string` | Vision backbone identifier (`dinov3-small`). |
| `aspect_mode` | `string` | Aspect ratio handling method (`approach_1_native_aspect_bicubic`). |
| `arithmetic_precision` | `string` | Precision guarantee (`float32` throughout). |
| `resampling_method` | `string` | Downsampling protocol: `srgb_to_linear_rgb_scale_to_srgb_to_cielab`. |
| `backbone_details.model_id` | `string` | HuggingFace repository ID for the base model. |
| `backbone_details.patch_size` | `int` | ViT patch edge in pixels (`16`). |
| `backbone_details.token_start_idx` | `int` | Token start index (`5` = skips 1 CLS token + 4 register tokens). |
| `in_dim` | `int` | Input token feature dimension (`768` = $2 \times 384$). |
| `hidden_dim` | `int` | Adapter internal bottleneck dimension (`384`). |
| `metrics.val_loss` | `float` | Best validation loss achieved during training. |
| `metrics.val_mae_L` | `float` | Mean absolute error on lightness ($L^*$). |
| `metrics.val_mae_ab` | `float` | Mean absolute error on chrominance ($a^*, b^*$). |
| `acceptance_gates` | `object` | Automated numerical quality pass/fail checks. |
