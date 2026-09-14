#!/usr/bin/env python3
"""DINOv3 + Photographic Intent Appearance Model Inversion Utility.

This script implements the self-learned, IT8-free film negative conversion pipeline
using DINOv3 vision backbone features and the trained Photographic Intent Adapter.

It directly leverages the optimized decoding, CFA calibration, and film base
formulations from negicc-station:
  - negicc_station.CapturedImage: High-performance C++/LibRaw demosaicing
  - film_profiling.FilmProfile: Profile metadata and reference film base extraction
  - crosstalk_calibration.apply_correction: CFA sensor de-crosstalk (I_cfa = I_sensor * M_cfa^T)
  - Transmittance & Optical Density:
      The profile's film base is measured at the reference exposure (1/8s, ISO 100).
      eff_base = max(base_cfa, percentile_99.8(I_cfa))
      T = clip(I_cfa / eff_base, 1e-4, 1.0)
      D = -log10(T)
  - Two-Stage Cascade Inversion:
      Stage 1 (Pass 0 Bootstrap): Fast 1D Photoshop Auto-WB curve hillclimb on preview
      Stage 2 (Physical Sensitometric Solver): Fixed-point Hurter & Driffield curve solver
        guided by DINOv3 CIELAB intent field (mu, sigma)

Usage:
  ./venv/bin/python3 src/appearance_model_film_conversion.py --raw sample.ARW --output output.jpg
"""

import os
import sys
import time
import json
import argparse
from typing import Optional, Tuple, Dict, Any, Union

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# Set multi-threading for PyTorch CPU operations
torch.set_num_threads(16)

# Ensure src directory is in path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import negicc_station
from film_profiling import FilmProfile, compute_exposure_ratio, parse_shutter_speed
from crosstalk_calibration import apply_correction

# ImageNet normalization tensors for DINOv3
MEAN_TENSOR = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_TENSOR = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DEFAULT_CROSSTALK_PROFILE = os.path.join(
    PROJECT_DIR, "profiles", "ILCE-7RM4_crosstalk_profile.json"
)
DEFAULT_FILM_PROFILE = os.path.join(
    PROJECT_DIR, "profiles", "profile_Portra400_20260623_170610.json"
)


# =============================================================================
# 1. Transmittance & Optical Density using main's CFA & Base formulations
# =============================================================================

def compute_negative_transmittance_and_density(
    img: Union[negicc_station.CapturedImage, str],
    profile: Optional[Union[FilmProfile, str, dict]] = None,
    film_base_rgb: Optional[Tuple[float, float, float]] = None,
    film_base_img: Optional[Union[negicc_station.CapturedImage, str]] = None,
    half: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Computes transmittance T and optical density D = -log10(T) using main's modules.
    
    Args:
        img: CapturedImage instance or path to RAW file.
        profile: FilmProfile instance, JSON path, or dict (defaults to Portra 400 profile).
        film_base_rgb: Optional custom film base RGB (r, g, b).
        film_base_img: Optional CapturedImage of unexposed film base.
        half: Whether to decode at half resolution (preview) or full resolution.
        
    Returns:
        transmittance: np.ndarray [H, W, 3] in [1e-4, 1.0]
        density: np.ndarray [H, W, 3] (D = -log10(T))
        meta: dict with anchor points and effective base vectors
    """
    # 1. Load CapturedImage
    if isinstance(img, str):
        img_cap = negicc_station.CapturedImage(0, 0.125, 100, [img])
    else:
        img_cap = img

    # Load profile instance if provided or use default
    prof_obj = None
    if profile is not None:
        if isinstance(profile, FilmProfile):
            prof_obj = profile
        elif isinstance(profile, (str, dict)):
            prof_obj = FilmProfile(profile)
    elif os.path.exists(DEFAULT_FILM_PROFILE):
        prof_obj = FilmProfile(DEFAULT_FILM_PROFILE)

    # 2. Extract linear sensor RGB buffer via C++ LibRaw
    raw_u16 = img_cap.to_numpy(half=half, pipeline="cpp", to_uint8=False, output_profile_path="linear")
    raw_float = raw_u16.astype(np.float32) / 65535.0

    # 3. Apply CFA de-crosstalk matrix via crosstalk_calibration.apply_correction
    if prof_obj is not None and hasattr(prof_obj, "crosstalk_matrix"):
        cc_matrix = prof_obj.crosstalk_matrix
    elif os.path.exists(DEFAULT_CROSSTALK_PROFILE):
        with open(DEFAULT_CROSSTALK_PROFILE, "r") as f:
            ct_data = json.load(f)
        cc_matrix = np.array(ct_data.get("crosstalk_correction_matrix", np.eye(3)), dtype=np.float64)
    else:
        cc_matrix = np.eye(3, dtype=np.float64)

    cfa_rgb = apply_correction(raw_float, cc_matrix).astype(np.float32)

    # 4. Determine film base RGB, shutter speed, and ISO
    fb_cfa = None
    fb_cap = None

    if film_base_img is not None:
        if isinstance(film_base_img, str):
            fb_cap = negicc_station.CapturedImage(0, 0.125, 100, [film_base_img])
        else:
            fb_cap = film_base_img
        fb_raw = fb_cap.to_numpy(half=half, pipeline="cpp", to_uint8=False, output_profile_path="linear")
        fb_float = fb_raw.astype(np.float32) / 65535.0
        fb_cfa = apply_correction(fb_float, cc_matrix).mean(axis=(0, 1))

    elif film_base_rgb is not None:
        fb_vec = np.array(film_base_rgb, dtype=np.float32)
        if fb_vec.max() > 1.0:
            fb_vec /= 65535.0
        fb_cfa = apply_correction(fb_vec.reshape(1, 1, 3), cc_matrix).squeeze()

    elif prof_obj is not None:
        p_fb_r, p_fb_g, p_fb_b = prof_obj.get_film_base_rgb()
        fb_vec = np.array([p_fb_r, p_fb_g, p_fb_b], dtype=np.float32) / 65535.0
        fb_cfa = apply_correction(fb_vec.reshape(1, 1, 3), cc_matrix).squeeze()

    # Fallback to estimating unexposed base from negative highlights
    if fb_cfa is None or np.all(fb_cfa <= 0):
        fb_cfa = np.percentile(cfa_rgb, 99.8, axis=(0, 1))

    # 5. Compute exposure ratio relative to film base via shared utility
    exposure_ratio = compute_exposure_ratio(
        img=img_cap,
        profile=prof_obj,
        film_base_img=fb_cap
    )

    # Base level aligned to current scan exposure
    base_aligned = fb_cfa / exposure_ratio

    # 6. Effective base safeguard (ensures T <= 1.0 everywhere)
    peak_crop = np.percentile(cfa_rgb, 99.8, axis=(0, 1))
    eff_base = np.maximum(base_aligned, peak_crop)

    # 7. Transmittance T and Optical Density D = -log10(T)
    transmittance = np.clip(cfa_rgb / (eff_base[np.newaxis, np.newaxis, :] + 1e-6), 1e-4, 1.0)
    density = -np.log10(transmittance)

    d_base = np.percentile(density, 0.2, axis=(0, 1)).astype(np.float32)
    d_sum = density.sum(axis=-1)
    hi_mask = d_sum >= np.percentile(d_sum, 99.5)
    if np.any(hi_mask):
        d_white = np.median(density[hi_mask], axis=0).astype(np.float32)
    else:
        d_white = np.percentile(density, 99.5, axis=(0, 1)).astype(np.float32)

    meta = {
        "eff_base": eff_base,
        "raw_float": raw_float,
        "cfa_rgb": cfa_rgb,
        "d_base": d_base,
        "d_white": d_white,
        "crosstalk_matrix": cc_matrix,
    }
    return transmittance, density, meta


# =============================================================================
# 2. Color Science Utilities (IEC 61966-2-1 sRGB & CIE 1976 CIELAB)
# =============================================================================

def linear_to_srgb_torch(lin: torch.Tensor) -> torch.Tensor:
    """Exact IEC 61966-2-1 piecewise sRGB transfer function in PyTorch."""
    lin = torch.clamp(lin, 0.0, 1.0)
    mask = lin <= 0.0031308
    return torch.where(
        mask,
        lin * 12.92,
        1.055 * (torch.clamp(lin, min=1e-7) ** (1.0 / 2.4)) - 0.055,
    )


def srgb_to_lab_torch(srgb: torch.Tensor) -> torch.Tensor:
    """Continuous float32 conversion from sRGB [0, 1] to CIE 1976 CIELAB (D65)."""
    srgb = torch.clamp(srgb, 1e-6, 1.0)
    mask = srgb <= 0.04045
    lin = torch.where(mask, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0:1], lin[..., 1:2], lin[..., 2:3]

    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    xr = X / 0.95047
    yr = Y / 1.00000
    zr = Z / 1.08883

    delta = 6.0 / 29.0
    d3 = delta ** 3
    c1 = 1.0 / (3.0 * delta ** 2)
    c2 = 4.0 / 29.0

    def f(t):
        return torch.where(t > d3, torch.clamp(t, min=1e-7) ** (1.0 / 3.0), c1 * t + c2)

    L = 116.0 * f(yr) - 16.0
    a = 500.0 * (f(xr) - f(yr))
    b_val = 200.0 * (f(yr) - f(zr))
    return torch.cat([L, a, b_val], dim=-1)


def compute_aspect_preserved_shape(w: int, h: int, max_size: int = 512, patch_size: int = 16) -> Tuple[int, int, int, int]:
    """Computes isotropic patch-aligned grid preserving native aspect ratio."""
    scale = max_size / float(max(w, h))
    target_w = max(patch_size, int(round((w * scale) / patch_size)) * patch_size)
    target_h = max(patch_size, int(round((h * scale) / patch_size)) * patch_size)
    grid_w = target_w // patch_size
    grid_h = target_h // patch_size
    return target_w, target_h, grid_w, grid_h


# =============================================================================
# 3. DINOv3 Photographic Intent Adapter & Model Loader
# =============================================================================

class PhotographicIntentAdapter(nn.Module):
    """Residual bottleneck MLP adapter mapping multi-layer vision tokens to CIELAB."""
    def __init__(self, in_dim: int = 768, hidden_dim: int = 384):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()

        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mean_head = nn.Linear(hidden_dim, 3)
        self.var_head = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.mean_head.weight)
        self.mean_head.bias.data = torch.tensor([50.0, 0.0, 0.0], dtype=torch.float32)

        nn.init.normal_(self.var_head.weight, std=0.02)
        self.var_head.bias.data = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.dtype != torch.float32:
            z = z.float()
        h = self.act1(self.norm1(self.in_proj(z)))
        h = self.norm2(h + self.res_block(h))
        mu = self.mean_head(h)
        log_var = torch.clamp(self.var_head(h), min=-2.5, max=5.0)
        sigma = torch.exp(0.5 * log_var)
        return mu, sigma, log_var


class DINOv3AppearanceModel:
    """Encapsulates the pre-trained DINOv3 foundation backbone and intent adapter."""
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_dir = model_dir

        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}

        self.model_name = meta.get("model_name", os.path.basename(model_dir))
        self.patch_size = meta.get("patch_size", 16)
        self.image_size = meta.get("image_size", 512)
        self.k_start = meta.get("k_start", 5)
        in_dim = meta.get("in_dim", 768)
        hidden_dim = meta.get("hidden_dim", 384)

        backbone_id = meta.get("backbone", "Tooony133/dinov3-vits16-pretrain-lvd1689m")
        local_backbones = [
            os.path.join(PROJECT_DIR, "models", "dinov3-small"),
            os.path.join(model_dir, "backbone"),
            backbone_id,
        ]
        load_path = backbone_id
        for candidate in local_backbones:
            if os.path.isdir(candidate) and (
                os.path.isfile(os.path.join(candidate, "config.json")) or
                os.path.isfile(os.path.join(candidate, "model.safetensors"))
            ):
                load_path = candidate
                break

        print(f"[DINOv3] Loading backbone from '{load_path}' on {self.device}...")
        self.dino = AutoModel.from_pretrained(load_path).to(self.device).eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        num_layers = getattr(self.dino.config, "num_hidden_layers", 12)
        self.l_photo = int(round(num_layers * 0.75))
        self.l_sem = num_layers

        reg_tokens = getattr(self.dino.config, "num_register_tokens", 0)
        if reg_tokens > 0:
            self.k_start = 1 + reg_tokens

        self.adapter = PhotographicIntentAdapter(in_dim=in_dim, hidden_dim=hidden_dim).to(self.device).eval()
        weights_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        ckpt = torch.load(weights_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.adapter.load_state_dict(state_dict)
        print(f"[DINOv3] Model ready! ({self.model_name}, Layers {self.l_photo}+{self.l_sem})")

    def predict_intent_field(self, preview_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Predicts spatial CIELAB intent field (mu, sigma) from preview image."""
        h_orig, w_orig = preview_u8.shape[:2]
        target_w, target_h, grid_w, grid_h = compute_aspect_preserved_shape(
            w_orig, h_orig, max_size=self.image_size, patch_size=self.patch_size
        )
        im = Image.fromarray(preview_u8).resize((target_w, target_h), Image.Resampling.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0
        t = (torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) - MEAN_TENSOR) / STD_TENSOR

        with torch.no_grad():
            outputs = self.dino(pixel_values=t.to(self.device), output_hidden_states=True)
            h_photo = outputs.hidden_states[self.l_photo][:, self.k_start:, :]
            h_sem = outputs.hidden_states[self.l_sem][:, self.k_start:, :]
            feats = torch.cat([h_photo, h_sem], dim=-1)
            mu, sigma, _ = self.adapter(feats)

        return (
            mu.squeeze(0).cpu().numpy(),
            sigma.squeeze(0).cpu().numpy(),
            (grid_h, grid_w),
            (target_w, target_h),
        )


# =============================================================================
# 4. Parametric Inversion Engines
# =============================================================================

class PhotoshopAutoWBInverter(nn.Module):
    """Stage 1: 1D Photoshop Auto-WB & Contrast Curve Engine with bounded parameters."""
    def __init__(self, fixed_bp: list, fixed_wp: list):
        super().__init__()
        self.register_buffer("bp", torch.tensor(fixed_bp, dtype=torch.float32))
        self.register_buffer("wp", torch.tensor(fixed_wp, dtype=torch.float32))
        self.raw_gamma = nn.Parameter(torch.zeros(3))
        self.raw_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, raw_inv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.clamp((raw_inv - self.bp) / (self.wp - self.bp + 1e-6), 0.0, 1.0)
        gamma = 0.70 + 1.20 * torch.sigmoid(self.raw_gamma)   # Bounded [0.70, 1.90]
        mid = torch.pow(torch.clamp(x, min=1e-6), gamma)
        s = 0.35 * torch.tanh(self.raw_s)
        scurve = mid + s * torch.sin(2.0 * np.pi * mid) / (2.0 * np.pi)
        srgb = torch.clamp(scurve, 0.0, 1.0)
        lab = srgb_to_lab_torch(srgb)
        return srgb, lab, gamma, s


class PhysicalFilmInverter(nn.Module):
    """Stage 2: Parametric Hurter & Driffield Physical Inversion Engine."""
    def __init__(
        self,
        d_base: list,
        d_white: list,
        init_gamma: list = [2.1, 2.3, 2.3],
        init_ped: list = [0.035, 0.025, 0.020],
    ):
        super().__init__()
        self.register_buffer("dmin", torch.tensor(d_base, dtype=torch.float32))
        self.register_buffer("dmax", torch.tensor(d_white, dtype=torch.float32))
        self.raw_gamma = nn.Parameter(torch.tensor([np.log(g) for g in init_gamma], dtype=torch.float32))
        self.raw_pedestal = nn.Parameter(torch.tensor([np.log(p) for p in init_ped], dtype=torch.float32))
        self.raw_gains = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        self.off_diag = nn.Parameter(torch.zeros(3, 3))

    def get_dye_matrix(self) -> torch.Tensor:
        mask = 1.0 - torch.eye(3, device=self.dmin.device)
        return torch.eye(3, device=self.dmin.device) + torch.tanh(self.off_diag) * mask * 0.15

    def forward(self, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = torch.clamp((D - self.dmin) / (self.dmax - self.dmin + 1e-6), 0.0, 1.0)
        gamma = torch.exp(self.raw_gamma)
        pedestal = torch.exp(self.raw_pedestal)
        H = pedestal + (1.0 - pedestal) * torch.pow(u, gamma)
        gains = torch.exp(self.raw_gains)
        E = H * gains
        M = self.get_dye_matrix()
        lin = torch.clamp(torch.matmul(E, M.T), 0.0, 1.0)
        srgb = linear_to_srgb_torch(lin)
        lab = srgb_to_lab_torch(srgb)
        return srgb, lab, lin, H


# =============================================================================
# 5. Conversion Pipeline & Optimization Loop
# =============================================================================

def convert_film_negative(
    raw_path: str,
    film_base_path: Optional[str] = None,
    profile_path: Optional[str] = None,
    model_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    full_res: bool = False,
    max_iters: int = 5,
    tolerance_delta_mu: float = 0.80,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs end-to-end negative conversion with DINOv3 guidance using main's modules."""
    t_start = time.time()
    basename = os.path.splitext(os.path.basename(raw_path))[0]

    if model_dir is None:
        model_dir = os.path.join(PROJECT_DIR, "models", "dinov3-small_portra400")

    print(f"\n================================================================================")
    print(f"  DINOv3 + Photographic Intent Film Negative Inversion")
    print(f"  Target: {basename} | Mode: {'Full-Res' if full_res else 'Half-Res'}")
    print(f"  Model:  {os.path.basename(model_dir)}")
    print(f"================================================================================")

    # 1. Compute Transmittance & Density using main's modules (negicc_station + film_profiling)
    print(f"\n[Transmittance] Computing optical density via negicc_station & film_profiling...")
    transmittance, density, meta = compute_negative_transmittance_and_density(
        img=raw_path,
        profile=profile_path,
        film_base_img=film_base_path,
        half=not full_res,
    )

    D_full = density
    h_orig, w_orig = D_full.shape[:2]
    d_base = meta["d_base"].tolist()
    d_white = meta["d_white"].tolist()
    eff_base = meta["eff_base"]
    print(f"  Resolution: {w_orig}x{h_orig} | Effective Base: R={eff_base[0]:.4f}, G={eff_base[1]:.4f}, B={eff_base[2]:.4f}")
    print(f"  Base Density:  R={d_base[0]:.4f}, G={d_base[1]:.4f}, B={d_base[2]:.4f}")
    print(f"  White Density: R={d_white[0]:.4f}, G={d_white[1]:.4f}, B={d_white[2]:.4f}")

    # 2. Load DINOv3 model
    guidance = DINOv3AppearanceModel(model_dir, device=device)

    # Prepare preview resolution
    target_w, target_h, grid_w, grid_h = compute_aspect_preserved_shape(
        w_orig, h_orig, max_size=guidance.image_size, patch_size=guidance.patch_size
    )
    gh, gw, ps = grid_h, grid_w, guidance.patch_size

    # Resize density to preview
    D_prev_np = cv2.resize(D_full, (target_w, target_h), interpolation=cv2.INTER_AREA)
    D_prev = torch.tensor(D_prev_np, dtype=torch.float32)

    # Raw inverted preview for Stage 1 Photoshop Auto-WB
    raw_rgb_prev = cv2.resize(meta["raw_float"], (target_w, target_h), interpolation=cv2.INTER_AREA)
    raw_norm = raw_rgb_prev / (np.max(raw_rgb_prev) + 1e-6)
    raw_inv_prev = torch.tensor(1.0 - raw_norm, dtype=torch.float32)

    inv_np = raw_inv_prev.numpy()
    fixed_bp = np.percentile(inv_np, 0.5, axis=(0, 1)).tolist()
    fixed_wp = np.percentile(inv_np, 99.5, axis=(0, 1)).tolist()

    # -------------------------------------------------------------------------
    # Stage 1: Fast Photoshop Auto-WB Bootstrap
    # -------------------------------------------------------------------------
    print(f"\n[Stage 1] Hillclimbing Photoshop Auto-WB Preview Curve...")
    ps_model = PhotoshopAutoWBInverter(fixed_bp, fixed_wp)
    ps_opt = torch.optim.Adam(ps_model.parameters(), lr=0.03)

    with torch.no_grad():
        init_srgb_ps, _, _, _ = ps_model(raw_inv_prev)
        init_ps_u8 = (init_srgb_ps.numpy() * 255.0).astype(np.uint8)
    tgt_mu_ps, _, _, _ = guidance.predict_intent_field(init_ps_u8)

    t0_st1 = time.time()
    for it in range(1, 4):
        tgt_t = torch.tensor(tgt_mu_ps, dtype=torch.float32)
        for step in range(80):
            ps_opt.zero_grad()
            srgb_p, lab_p, _, _ = ps_model(raw_inv_prev)
            lab_patches = lab_p.view(gh, ps, gw, ps, 3).permute(0, 2, 1, 3, 4).contiguous().view(gh * gw, ps * ps, 3)
            pred_mu = torch.mean(lab_patches, dim=1)
            diff = pred_mu - tgt_t
            loss = (
                F.smooth_l1_loss(diff[:, 0], torch.zeros_like(diff[:, 0]), beta=1.0)
                + 3.0 * F.smooth_l1_loss(diff[:, 1], torch.zeros_like(diff[:, 1]), beta=1.0)
                + 3.0 * F.smooth_l1_loss(diff[:, 2], torch.zeros_like(diff[:, 2]), beta=1.0)
            )
            loss.backward()
            ps_opt.step()

        with torch.no_grad():
            srgb_eval, _, _, _ = ps_model(raw_inv_prev)
            curr_u8 = (torch.clamp(srgb_eval, 0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            new_mu, _, _, _ = guidance.predict_intent_field(curr_u8)
            delta_mu = float(np.mean(np.sqrt(np.sum((new_mu - tgt_mu_ps) ** 2, axis=-1))))
            tgt_mu_ps = 0.5 * tgt_mu_ps + 0.5 * new_mu

        if delta_mu < 0.85:
            break

    print(f"  Stage 1 converged in {time.time() - t0_st1:.2f}s (Delta E: {delta_mu:.3f})")

    # Generate Stage 1 bootstrap preview for Stage 2
    with torch.no_grad():
        srgb_st1, _, _, _ = ps_model(raw_inv_prev)
        img_stage1_u8 = (torch.clamp(srgb_st1, 0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    tgt_mu_st2, _, _, _ = guidance.predict_intent_field(img_stage1_u8)

    # -------------------------------------------------------------------------
    # Stage 2: Physical Sensitometric Fixed-Point Solver
    # -------------------------------------------------------------------------
    print(f"\n[Stage 2] Physical Sensitometric Fixed-Point Inversion...")
    stage2_model = PhysicalFilmInverter(
        d_base=d_base,
        d_white=d_white,
        init_gamma=[2.1, 2.3, 2.3],
        init_ped=[0.035, 0.025, 0.020],
    )
    stage2_opt = torch.optim.Adam(stage2_model.parameters(), lr=0.02)

    history = []
    t0_st2 = time.time()
    for it in range(1, max_iters + 1):
        t_it0 = time.time()
        tgt_t = torch.tensor(tgt_mu_st2, dtype=torch.float32)

        for step in range(100):
            stage2_opt.zero_grad()
            srgb_p, lab_p, _, _ = stage2_model(D_prev)
            lab_patches = lab_p.view(gh, ps, gw, ps, 3).permute(0, 2, 1, 3, 4).contiguous().view(gh * gw, ps * ps, 3)
            pred_mu = torch.mean(lab_patches, dim=1)
            diff = pred_mu - tgt_t
            loss = (
                F.smooth_l1_loss(diff[:, 0], torch.zeros_like(diff[:, 0]), beta=1.0)
                + 4.0 * F.smooth_l1_loss(diff[:, 1], torch.zeros_like(diff[:, 1]), beta=1.0)
                + 4.0 * F.smooth_l1_loss(diff[:, 2], torch.zeros_like(diff[:, 2]), beta=1.0)
                + 0.01 * torch.sum(stage2_model.off_diag ** 2)
            )
            loss.backward()
            stage2_opt.step()

        with torch.no_grad():
            srgb_eval, _, _, _ = stage2_model(D_prev)
            curr_u8 = (torch.clamp(srgb_eval, 0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            new_mu, _, _, _ = guidance.predict_intent_field(curr_u8)
            delta_mu = float(np.mean(np.sqrt(np.sum((new_mu - tgt_mu_st2) ** 2, axis=-1))))
            tgt_mu_st2 = 0.5 * tgt_mu_st2 + 0.5 * new_mu

        dur = time.time() - t_it0
        history.append({"iter": it, "delta_mu": delta_mu, "time": dur})
        print(f"  Iteration {it}/{max_iters} | Delta E: {delta_mu:.3f} dE | Time: {dur:.2f}s")
        if delta_mu < tolerance_delta_mu:
            print(f"  Converged within tolerance ({tolerance_delta_mu} dE) at iteration {it}!")
            break

    total_solve_time = time.time() - t0_st2
    print(f"  Stage 2 solved in {total_solve_time:.2f}s")

    # -------------------------------------------------------------------------
    # 5. Apply Converged Parameters to Target Resolution
    # -------------------------------------------------------------------------
    print(f"\n[Inversion] Applying converged physical parameters to negative density...")
    with torch.no_grad():
        D_target_t = torch.tensor(D_full, dtype=torch.float32)
        srgb_out, _, _, _ = stage2_model(D_target_t)
        output_u8 = (torch.clamp(srgb_out, 0.0, 1.0).numpy() * 255.0).astype(np.uint8)

    # Save output if specified
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            Image.fromarray(output_u8).save(output_path, quality=95)
        elif ext == ".png":
            Image.fromarray(output_u8).save(output_path)
        elif ext in (".tif", ".tiff"):
            Image.fromarray(output_u8).save(output_path, compression="tiff_lzw")
        print(f"[Output] Saved positive image to: {output_path}")

    total_duration = time.time() - t_start
    print(f"[Finished] Full conversion completed in {total_duration:.2f}s!\n")

    info = {
        "basename": basename,
        "resolution": [w_orig, h_orig],
        "iterations": len(history),
        "final_delta_mu": history[-1]["delta_mu"] if history else 0.0,
        "total_time_seconds": total_duration,
        "eff_base": eff_base.tolist(),
        "d_base": d_base,
        "d_white": d_white,
    }
    return output_u8, info


def convert_captured_image_learned(
    img: Any,
    film_base: Optional[Any] = None,
    profile_path: Optional[str] = None,
    model_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    half: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Programmatic Python API for converting a CapturedImage using learned appearance."""
    return convert_film_negative(
        raw_path=img if isinstance(img, str) else img.filepaths[0],
        film_base_path=film_base if isinstance(film_base, str) else (film_base.filepaths[0] if film_base else None),
        profile_path=profile_path,
        model_dir=model_dir,
        output_path=output_path,
        full_res=not half,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert film negative RAW image using DINOv3 + Photographic Intent Field appearance model."
    )
    parser.add_argument("--raw", required=True, help="Path to digitized negative RAW image (.ARW, .dng, etc.)")
    parser.add_argument("--base", default=None, help="Optional path to unexposed film base RAW image")
    parser.add_argument("--profile", default=None, help="Optional path to film profile JSON")
    parser.add_argument("--model", default=None, help="Path to trained appearance model directory (default: models/dinov3-small_portra400)")
    parser.add_argument("--output", default="build/converted_positive.jpg", help="Output positive image path (.jpg, .png, .tiff)")
    parser.add_argument("--full", action="store_true", help="Process at full resolution instead of half resolution")
    parser.add_argument("--max-iters", type=int, default=5, help="Maximum fixed-point iterations (default: 5)")
    parser.add_argument("--tolerance", type=float, default=0.80, help="Delta E convergence threshold (default: 0.80)")
    parser.add_argument("--device", default="cpu", help="Compute device: cpu or cuda (default: cpu)")

    args = parser.parse_args()

    convert_film_negative(
        raw_path=args.raw,
        film_base_path=args.base,
        profile_path=args.profile,
        model_dir=args.model,
        output_path=args.output,
        full_res=args.full,
        max_iters=args.max_iters,
        tolerance_delta_mu=args.tolerance,
        device=args.device,
    )


if __name__ == "__main__":
    main()
