"""Native Aspect-Ratio Feature Extraction & Caching Pipeline for DINOv3.

Implements Two-Paths Architecture:
- Path A (Backbone): Standard sRGB bilinear resize -> ImageNet normalization -> ViT
- Path B (Target): Linear-light radiometric area box downsampling -> float32 CIELAB
- Multi-layer token fusion: Layer 9 (photographic) + Layer 12 (semantic)
- Dynamic register token slicing: discards first 5 tokens ([CLS] + 4 register tokens)
- Stores all features and targets in pure torch.float32 (zero quantization error)
"""

import os
import time
import numpy as np
import cv2
from PIL import Image
import torch

from models import BACKBONE_CONFIGS, load_backbone
from dataset import (
    load_profile_corpus,
    group_images_into_aspect_buckets,
    MEAN_TENSOR,
    STD_TENSOR,
)


def make_srgb_to_linear_lut():
    """Precomputes exact IEC 61966-2-1 inverse sRGB EOTF lookup table."""
    s = np.arange(256, dtype=np.float32) / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4).astype(np.float32)


SRGB_TO_LINEAR_LUT = make_srgb_to_linear_lut()


def linear_to_srgb(lin):
    """Converts linear RGB [0.0, 1.0] to sRGB float32."""
    l = np.clip(lin, 0.0, 1.0)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * (l ** (1.0 / 2.4)) - 0.055).astype(np.float32)


def resample_in_linear_light(im_u8, target_w, target_h):
    """Resamples image in linear physical energy: sRGB -> Linear RGB -> scale -> sRGB float32."""
    arr_lin = SRGB_TO_LINEAR_LUT[im_u8]
    h, w = im_u8.shape[:2]
    interp = cv2.INTER_AREA if (target_w <= w and target_h <= h) else cv2.INTER_LINEAR
    lin_scaled = cv2.resize(arr_lin, (target_w, target_h), interpolation=interp)
    srgb_scaled = linear_to_srgb(lin_scaled)
    return srgb_scaled


def extract_native_aspect_features(
    profile_name,
    backbone_name="dinov3-small",
    corpus_dir="/home/alpha/Projects/FilmCorpus/training",
    cache_dir="cache",
    max_samples=400,
    batch_size=16,
    force_extract=False,
    device="cpu",
):
    """Extracts and caches multi-layer patch features using native aspect grids and two-path linear targets."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{backbone_name}_{profile_name}_cache.pt")

    if os.path.exists(cache_path) and not force_extract:
        print(f"[{backbone_name} | {profile_name}] Loading cached features from {cache_path}...", flush=True)
        cache = torch.load(cache_path, map_location="cpu")
        print(f"  Features: {cache['features'].shape} ({cache['features'].dtype}), Targets: {cache['targets'].shape} ({cache['targets'].dtype})", flush=True)
        return cache["features"], cache["targets"], cache_path

    image_files = load_profile_corpus(profile_name, corpus_dir=corpus_dir, max_samples=max_samples)
    if not image_files:
        raise ValueError(f"No valid images found for profile '{profile_name}' in {corpus_dir}")

    cfg = BACKBONE_CONFIGS[backbone_name]
    max_size = cfg["image_size"]
    patch_size = cfg["patch_size"]
    k_start = cfg["token_start_idx"]

    print(f"\n================================================================================", flush=True)
    print(f" Extracting Multi-Layer DINOv3 Features: {backbone_name}", flush=True)
    print(f" Profile: {profile_name} | Images: {len(image_files)}", flush=True)
    print(f" Max Edge: {max_size}px | Patch Size: {patch_size}px | Token Start: {k_start}", flush=True)
    print(f" Precision: float32 throughout | Target Pipeline: Linear-Light Area Downsampling", flush=True)
    print(f"================================================================================\n", flush=True)

    model, _ = load_backbone(backbone_name=backbone_name, device=device)
    num_layers = getattr(model.config, "num_hidden_layers", 12)
    l_photo = int(round(num_layers * 0.75))  # Layer 9
    l_sem = num_layers                       # Layer 12

    reg_tokens = getattr(model.config, "num_register_tokens", 0)
    if reg_tokens > 0:
        k_start = 1 + reg_tokens

    buckets = group_images_into_aspect_buckets(image_files, max_size=max_size, patch_size=patch_size)
    print(f"Aspect Bucketing: Sorted {len(image_files)} images into {len(buckets)} native shape buckets.")

    all_feat_chunks = []
    all_tgt_chunks = []
    t0 = time.time()
    total_extracted_patches = 0

    mean_t = MEAN_TENSOR.to(device)
    std_t = STD_TENSOR.to(device)

    for b_idx, (b_shape, b_files) in enumerate(buckets.items(), start=1):
        target_w, target_h, grid_w, grid_h = b_shape
        N_patches = grid_w * grid_h
        print(f"  [Bucket {b_idx}/{len(buckets)}] {target_w}x{target_h} ({grid_w}x{grid_h} = {N_patches} patches) -> {len(b_files)} images", flush=True)

        for i in range(0, len(b_files), batch_size):
            chunk_files = b_files[i : i + batch_size]
            t_imgs = []
            targets_np = []

            for fpath in chunk_files:
                try:
                    with Image.open(fpath) as pil_im:
                        pil_rgb = pil_im.convert("RGB")
                        im_u8 = np.array(pil_rgb)

                    # Path A: sRGB bilinear resize for Vision Backbone
                    im_resized = pil_rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
                    arr = np.array(im_resized, dtype=np.float32) / 255.0
                    t_imgs.append(torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32))

                    # Path B: Linear-Light radiometric downsampling for CIELAB Target
                    srgb_scaled = resample_in_linear_light(im_u8, target_w, target_h)
                    lab = cv2.cvtColor(srgb_scaled, cv2.COLOR_RGB2LAB)
                    patch_lab = (
                        lab.reshape(grid_h, patch_size, grid_w, patch_size, 3)
                        .transpose(0, 2, 1, 3, 4)
                        .reshape(N_patches, patch_size, patch_size, 3)
                        .mean(axis=(1, 2))
                    )
                    targets_np.append(patch_lab)
                except Exception as e:
                    print(f"    Warning: Skipping {fpath}: {e}")
                    continue

            if not t_imgs:
                continue

            batch_t = (torch.stack(t_imgs).to(device) - mean_t) / std_t
            with torch.no_grad():
                out = model(pixel_values=batch_t, output_hidden_states=True)
                h_photo = out.hidden_states[l_photo][:, k_start:, :]
                h_sem = out.hidden_states[l_sem][:, k_start:, :]
                fused = torch.cat([h_photo, h_sem], dim=-1).cpu().float()

            B_curr = fused.shape[0]
            fused_flat = fused.reshape(B_curr * N_patches, fused.shape[-1])
            tgt_flat = torch.from_numpy(np.stack(targets_np)).float().reshape(B_curr * N_patches, 3)

            all_feat_chunks.append(fused_flat)
            all_tgt_chunks.append(tgt_flat)
            total_extracted_patches += B_curr * N_patches

    dur = time.time() - t0
    print(f"\nExtraction complete in {dur:.2f}s! Total Tokens: {total_extracted_patches:,} ({total_extracted_patches/dur:.1f} tokens/s).", flush=True)

    features_all = torch.cat(all_feat_chunks, dim=0)
    targets_all = torch.cat(all_tgt_chunks, dim=0)

    print(f"Final Tensor Dimensions: Features: {features_all.shape} ({features_all.dtype}), Targets: {targets_all.shape} ({targets_all.dtype})")
    torch.save({"features": features_all, "targets": targets_all}, cache_path)
    print(f"Saved feature cache to: {cache_path}")
    return features_all, targets_all, cache_path
