"""Neural Architecture definitions for DINOv3 Photographic Intent Field Appearance Models.

Implements the multi-layer dense spatial-semantic specification:
- DINOv3-small vision backbone (ViT-S/16, 512x512, 1,024 patches)
- Multi-layer extraction: L_photo (~0.75 L = Layer 9) + L_sem (Layer 12)
- Fused token dimension: 2 * 384 = 768-D
- Token start index: 5 (discards 1 [CLS] + 4 register tokens)
- PhotographicIntentAdapter: Residual bottleneck MLP with LayerNorm and GELU
- Physical prior initialization: mu_0 = [50.0, 0.0, 0.0], sigma_0 = [2.0, 1.0, 1.0]
- Log variance clamped to [-2.5, 5.0]
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModel

BACKBONE_CONFIGS = {
    "dinov3-small": {
        "model_id": "Tooony133/dinov3-vits16-pretrain-lvd1689m",
        "image_size": 512,
        "patch_size": 16,
        "grid_size": 32,
        "num_patches": 1024,
        "raw_dim": 384,
        "fused_dim": 768,
        "adapter_hidden_dim": 384,
        "token_start_idx": 5,   # 1 [CLS] + 4 register tokens
        "description": "Meta DINOv3 Small (ViT-S/16, 512x512, 1,024 patches, Layers 9+12 fused 768-D)",
    },
}


class PhotographicIntentAdapter(nn.Module):
    """Residual MLP adapter mapping multi-layer vision tokens to CIELAB (mu, sigma).
    
    Specification:
        h1 = GELU(LayerNorm(W_in * z + b_in))          [hidden_dim]
        h2 = LayerNorm(h1 + W2 * GELU(LayerNorm(W1 * h1 + b1)) + b2)
        mu = W_mu * h2 + b_mu                          [3] (L*, a*, b*)
        log_sigma2 = clamp(W_sigma * h2 + b_sigma, -2.5, 5.0)
        sigma = exp(0.5 * log_sigma2)
    """

    def __init__(self, in_dim=768, hidden_dim=384):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

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

        # Grounded physical priors
        # Mean head: neutral mid-gray (L*=50.0, a*=0.0, b*=0.0)
        nn.init.zeros_(self.mean_head.weight)
        self.mean_head.bias.data = torch.tensor([50.0, 0.0, 0.0], dtype=torch.float32)

        # Variance head: realistic initial standard deviations
        nn.init.normal_(self.var_head.weight, std=0.02)
        self.var_head.bias.data = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)

    def forward(self, z):
        """Forward pass for multi-layer patch tokens.
        
        Args:
            z: Tensor of shape [B, N_patches, in_dim]
            
        Returns:
            mu: Predicted CIELAB means [B, N_patches, 3]
            sigma: Predicted CIELAB standard deviations [B, N_patches, 3]
            log_var: Clamped log variances [B, N_patches, 3]
        """
        if z.dtype != torch.float32:
            z = z.float()

        h = self.act1(self.norm1(self.in_proj(z)))
        h = self.norm2(h + self.res_block(h))

        mu = self.mean_head(h)
        log_var = torch.clamp(self.var_head(h), min=-2.5, max=5.0)
        sigma = torch.exp(0.5 * log_var)
        return mu, sigma, log_var


def load_backbone(backbone_name="dinov3-small", local_dir=None, device="cpu"):
    """Loads frozen vision backbone. Checks local directory first before downloading."""
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choices: {list(BACKBONE_CONFIGS.keys())}")

    cfg = BACKBONE_CONFIGS[backbone_name]
    model_id = cfg["model_id"]

    # Check local path candidates
    candidates = []
    if local_dir:
        candidates.append(local_dir)
    # Check ../models/dinov3-small relative to training dir
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(cur_dir, "..", "models", "dinov3-small"))
    candidates.append(os.path.join(cur_dir, "models", "dinov3-small"))

    load_path = model_id
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")):
            load_path = c
            print(f"Found local DINOv3 backbone at {load_path}")
            break

    print(f"Loading backbone from '{load_path}' on {device}...")
    model = AutoModel.from_pretrained(load_path).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, cfg
