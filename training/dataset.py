"""Dataset loader, aspect-ratio bucketing, and ground-truth CIELAB extraction.

Implements:
- Native aspect-ratio preservation without anisotropic distortion or black padding
- Dynamic patch-aligned aspect bucketing: W_target x H_target -> grid_w x grid_h
- Exact isotropic spatial patches (16x16 for DINOv3)
- Ground-truth CIELAB extraction: L* in [0, 100], a*, b* in [-128, 127]
"""

import os
import glob
import json
import random
from PIL import Image
import torch

MEAN_TENSOR = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_TENSOR = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def compute_aspect_preserved_shape(w, h, max_size=512, patch_size=16):
    """Computes patch-aligned target dimensions preserving native aspect ratio.
    
    Args:
        w: Native image width
        h: Native image height
        max_size: Maximum edge budget (512 for DINOv3)
        patch_size: Patch size in pixels (16 for DINOv3)
        
    Returns:
        target_w, target_h, grid_w, grid_h
    """
    scale = max_size / float(max(w, h))
    target_w = max(patch_size, int(round((w * scale) / patch_size)) * patch_size)
    target_h = max(patch_size, int(round((h * scale) / patch_size)) * patch_size)
    grid_w = target_w // patch_size
    grid_h = target_h // patch_size
    return target_w, target_h, grid_w, grid_h


def load_profile_corpus(profile_name, corpus_dir="/home/alpha/Projects/FilmCorpus/training", max_samples=None, random_seed=42):
    """Loads validated image paths for a film profile from training metadata or directory."""
    corpus_dir = os.path.expanduser(corpus_dir)
    aliases = {"pro160": "pro160ns", "pro400": "pro400h"}
    canonical = aliases.get(profile_name.lower(), profile_name.lower())

    image_paths = []

    # Try 1: JSON file at <corpus_dir>/<canonical>.json
    json_path = os.path.join(corpus_dir, f"{canonical}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            rel = it.get("image_path")
            if rel:
                full = os.path.join(corpus_dir, rel)
                if os.path.isfile(full):
                    image_paths.append(full)

    # Try 2: Directory at <corpus_dir>/<canonical>/images/
    if not image_paths:
        img_dir = os.path.join(corpus_dir, canonical, "images")
        if os.path.isdir(img_dir):
            image_paths = sorted(glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(img_dir, "*.[pP][nN][gG]")))

    # Try 3: Direct folder of images
    if not image_paths:
        direct_dir = os.path.join(corpus_dir, canonical)
        if os.path.isdir(direct_dir):
            image_paths = sorted(glob.glob(os.path.join(direct_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(direct_dir, "*.[pP][nN][gG]")))

    # Try 4: Top-level corpus directory directly
    if not image_paths and os.path.isdir(corpus_dir):
        image_paths = sorted(glob.glob(os.path.join(corpus_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(corpus_dir, "*.[pP][nN][gG]")))

    if not image_paths:
        raise FileNotFoundError(f"Could not locate training images for profile '{profile_name}' in '{corpus_dir}'")

    print(f"[{profile_name}] Loaded {len(image_paths)} existing images from corpus.")

    if max_samples is not None and len(image_paths) > max_samples:
        rng = random.Random(random_seed)
        shuffled = list(image_paths)
        rng.shuffle(shuffled)
        image_paths = sorted(shuffled[:max_samples])
        print(f"[{profile_name}] Subsampled to {len(image_paths)} images (seed={random_seed}).")

    return image_paths


def group_images_into_aspect_buckets(image_paths, max_size=512, patch_size=16):
    """Groups image files by native patch-aligned aspect ratio bucket."""
    buckets = {}
    for fpath in image_paths:
        try:
            with Image.open(fpath) as img:
                w, h = img.size
            b_key = compute_aspect_preserved_shape(w, h, max_size=max_size, patch_size=patch_size)
        except Exception:
            b_key = (max_size, max_size, max_size // patch_size, max_size // patch_size)
        buckets.setdefault(b_key, []).append(fpath)
    return buckets
