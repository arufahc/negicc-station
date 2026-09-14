"""Model Exporter for Production Deployments.

Exports trained PhotographicIntentAdapter models into:
1. PyTorch checkpoint (model.pt): Weights, optimizer state, metrics
2. TorchScript (model.torchscript): Traced JIT module for C++ runtime
3. ONNX (model.onnx): Open Neural Network Exchange graph with dynamic batch/patch axes
4. Metadata JSON (metadata.json): Complete parameter and architecture documentation
"""

import os
import json
import datetime
import torch

from models import PhotographicIntentAdapter, BACKBONE_CONFIGS


def export_appearance_model(checkpoint_path, export_dir, dummy_batch_size=1):
    """Exports a trained model checkpoint to all standard deployment formats."""
    os.makedirs(export_dir, exist_ok=True)

    print(f"\nLoading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    backbone_name = ckpt.get("backbone", "dinov3-small")
    profile_name = ckpt.get("profile", "unknown")
    cfg = BACKBONE_CONFIGS.get(backbone_name, BACKBONE_CONFIGS["dinov3-small"])
    num_patches = cfg.get("num_patches", 1024)
    in_dim = ckpt.get("in_dim", cfg["fused_dim"])
    hidden_dim = ckpt.get("hidden_dim", cfg["adapter_hidden_dim"])

    # Load and initialize adapter
    adapter = PhotographicIntentAdapter(in_dim=in_dim, hidden_dim=hidden_dim)
    adapter.load_state_dict(ckpt["model_state_dict"])
    adapter.eval()

    param_count = sum(p.numel() for p in adapter.parameters())

    # 1. Save standard PyTorch weights
    target_pt = os.path.join(export_dir, "model.pt")
    if os.path.abspath(checkpoint_path) != os.path.abspath(target_pt):
        torch.save(ckpt, target_pt)
    print(f"  [1/4] PyTorch weights saved: {target_pt}")

    # 2. Export TorchScript (JIT trace with dynamic tokens)
    target_ts = os.path.join(export_dir, "model.torchscript")
    dummy_input = torch.randn(dummy_batch_size, num_patches, in_dim)
    with torch.no_grad():
        traced_model = torch.jit.trace(adapter, dummy_input)
        traced_model.save(target_ts)
    print(f"  [2/4] TorchScript module saved: {target_ts}")

    # 3. Export ONNX graph with dynamic sequence length (if available)
    target_onnx = os.path.join(export_dir, "model.onnx")
    try:
        with torch.no_grad():
            torch.onnx.export(
                adapter,
                dummy_input,
                target_onnx,
                input_names=["tokens"],
                output_names=["mu", "sigma", "log_var"],
                dynamic_axes={
                    "tokens": {0: "batch_size", 1: "num_patches"},
                    "mu": {0: "batch_size", 1: "num_patches"},
                    "sigma": {0: "batch_size", 1: "num_patches"},
                    "log_var": {0: "batch_size", 1: "num_patches"},
                },
                opset_version=18,
            )
        print(f"  [3/4] ONNX graph saved: {target_onnx}")
    except Exception as e:
        print(f"  [3/4] ONNX export skipped ({e})")

    # 4. Save metadata JSON
    metrics = ckpt.get("metrics", {})
    val_loss = metrics.get("val_loss", 0.0)
    val_mae_L = metrics.get("val_mae_L", 0.0)
    val_mae_ab = metrics.get("val_mae_ab", 0.0)

    # Acceptance Gates
    gates = {
        "gate_val_loss_le_2_30": val_loss <= 2.30,
        "gate_mae_ab_le_5_00": val_mae_ab <= 5.00,
        "gate_mae_L_le_8_00": val_mae_L <= 8.00,
    }
    all_passed = all(gates.values())

    target_meta = os.path.join(export_dir, "metadata.json")
    meta_info = {
        "model_name": f"{backbone_name}_{profile_name}_intent_field",
        "profile": profile_name,
        "backbone": backbone_name,
        "aspect_mode": "approach_1_native_aspect_bicubic",
        "is_native_aspect": True,
        "arithmetic_precision": "float32",
        "mantissa_bits": 24,
        "resampling_method": "srgb_to_linear_rgb_scale_to_srgb_to_cielab",
        "color_pipeline": "Exact IEC 61966-2-1 inverse EOTF -> cv2.INTER_AREA linear scale -> forward EOTF -> cv2.COLOR_RGB2LAB (float32)",
        "backbone_details": cfg,
        "parameters": param_count,
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "export_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_dimensions": {
            "batch_size": "variable",
            "num_patches": "variable_native_aspect (e.g. 992 for 3:2, 768 for 4:3, 1024 for 1:1)",
            "token_dim": in_dim,
        },
        "output_dimensions": {
            "mu": ["batch_size", "num_patches", 3],
            "sigma": ["batch_size", "num_patches", 3],
            "log_var": ["batch_size", "num_patches", 3],
        },
        "color_space": "CIE L*a*b*",
        "channel_ranges": {
            "L*": [0.0, 100.0],
            "a*": [-128.0, 127.0],
            "b*": [-128.0, 127.0],
        },
        "metrics": metrics,
        "acceptance_gates": {
            "gates": gates,
            "all_passed": all_passed,
        },
        "exported_files": {
            "pytorch_weights": "model.pt",
            "torchscript": "model.torchscript",
            "onnx": "model.onnx",
            "metadata": "metadata.json",
        },
    }

    with open(target_meta, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    print(f"  [4/4] Metadata JSON saved: {target_meta}")

    # 5. Write README.md in export dir
    readme_path = os.path.join(export_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# Photographic Intent Field: {profile_name.upper()} ({backbone_name})\n\n")
        f.write(f"This model directory contains the trained appearance model for **{profile_name.upper()}** using **{backbone_name}**.\n\n")
        f.write("## Exported Files\n\n")
        f.write("- `model.pt`: Full PyTorch checkpoint with model weights and training metadata.\n")
        f.write("- `model.torchscript`: Traced TorchScript module for standalone C++ LibTorch execution.\n")
        f.write("- `model.onnx` / `model.onnx.data`: ONNX graph with dynamic batch and patch dimensions.\n")
        f.write("- `metadata.json`: Complete specification of backbone configuration, patch dimensions, and metrics.\n\n")
        f.write("## Validation Metrics\n\n")
        f.write(f"- **Validation Loss:** {val_loss:.4f}\n")
        f.write(f"- **MAE L*:** {val_mae_L:.2f}\n")
        f.write(f"- **MAE ab*:** {val_mae_ab:.2f}\n")
        f.write(f"- **All Gates Passed:** {'YES' if all_passed else 'NO'}\n")
    print(f"  README saved: {readme_path}")
    return meta_info
