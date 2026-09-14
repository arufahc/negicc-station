# Photographic Intent Field Model: Kodak Portra 400 (DINOv3-Small)

## 1. Specifications & Architecture
- **Backbone**: `Tooony133/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16, 21M parameters)
- **Aspect Mode**: Approach 1 Native Aspect-Ratio Grids with 2D Bicubic Positional Interpolation
- **Input Dimensions**: Dynamic patch-aligned aspect grids (e.g. $512 \times 336 \to 32 \times 21 = 672$ tokens; $512 \times 512 \to 32 \times 32 = 1{,}024$ tokens)
- **Patch Size**: $16 \times 16$ pixels
- **Register Tokens**: 4 register tokens + 1 CLS token correctly discarded ($k_{\\text{start}} = 5$)
- **Token Fusion**: Layers 9 ($L_{\\text{photo}}$) + 12 ($L_{\\text{sem}}$) concatenated $\\to 768\\text{-D}$ fused tokens
- **Resampling Pipeline**: Exact IEC 61966-2-1 inverse EOTF $\\to$ `cv2.INTER_AREA` linear scaling $\\to$ forward EOTF $\\to$ `cv2.COLOR_RGB2LAB` (zero gamma downsampling error)
- **Precision**: Continuous IEEE 754 32-bit floating point (`torch.float32`, 24 mantissa bits) throughout
- **Adapter**: Residual bottleneck MLP ($768 \\to 384 \\to 384 \\to 3$)
- **Total Adapter Parameters**: 595,590
- **Training Corpus**: 400 authentic Kodak Portra 400 laboratory master scans ([`~/Projects/FilmCorpus/training/portra400/images`](file:///home/alpha/Projects/FilmCorpus/training/portra400/images))
- **Design Reference**: [`~/Projects/negicc-station/designs/training_procedure_appearance_model.md`](file:///home/alpha/Projects/negicc-station/designs/training_procedure_appearance_model.md)

## 2. Validation Metrics & Acceptance Gates (60 Epochs)
- **Validation Huber Loss**: `1.7646` (Gate $\\le 2.30$: **PASSED**)
- **Validation MAE Total**: `4.13` CIELAB units
- **Validation MAE $L^*$**: `3.75` CIELAB units (Gate $\\le 8.00$: **PASSED**)
- **Validation MAE $a^*, b^*$**: `4.32` CIELAB units (Gate $\\le 5.00$: **PASSED**)
- **Uncertainty Dynamic Range**: `42.52x` (Gate $\\ge 8.0\\times$: **PASSED**)
- **Acceptance Gate Status**: **ALL GATES PASSED**

## 3. Implementation Code References
- **Adapter Architecture**: [`~/Projects/FilmModels/src/models.py`](file:///home/alpha/Projects/FilmModels/src/models.py)
- **Loss Function**: [`~/Projects/FilmModels/src/loss.py`](file:///home/alpha/Projects/FilmModels/src/loss.py)
- **Dataset & Aspect Bucketing**: [`~/Projects/FilmModels/src/dataset.py`](file:///home/alpha/Projects/FilmModels/src/dataset.py)
- **Feature Extraction & Linear Resampling**: [`~/Projects/FilmModels/src/extract_features.py`](file:///home/alpha/Projects/FilmModels/src/extract_features.py)
- **Training Loop**: [`~/Projects/FilmModels/src/train.py`](file:///home/alpha/Projects/FilmModels/src/train.py)
- **Model Exporter**: [`~/Projects/FilmModels/src/export.py`](file:///home/alpha/Projects/FilmModels/src/export.py)
- **Inference Pipeline**: [`~/Projects/FilmModels/src/infer.py`](file:///home/alpha/Projects/FilmModels/src/infer.py)

## 4. Exported Files
- `model.pt`: PyTorch weights and training metadata
- `model.torchscript`: Traced TorchScript module for C++ libtorch runtime
- `model.onnx` & `model.onnx.data`: ONNX graph with dynamic sequence length axes
- `metadata.json`: Full machine-readable specification, hyperparameters, and metrics
