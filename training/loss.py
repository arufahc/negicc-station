"""Heteroscedastic Robust Huber Loss Formulation for Photographic Intent Fields.

Standardized residual:
    r_i = |y_i - mu_i| / sigma_i
    H(r) = 0.5 * r^2 if r < delta else delta * (r - 0.5 * delta)
    Loss = mean(H(r) + 0.5 * log(sigma^2))
"""

import torch
import torch.nn as nn


class HeteroscedasticHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, mu, sigma, log_var, target):
        """Computes the heteroscedastic Huber loss on standardized residuals.
        
        Args:
            mu: Predicted mean tensor [B, N, 3] (L*, a*, b*)
            sigma: Predicted standard deviation tensor [B, N, 3]
            log_var: Clamped log variance tensor [B, N, 3]
            target: Ground truth CIELAB patch tensor [B, N, 3]
            
        Returns:
            loss: Scalar loss value
            metrics: Dict containing decomposed MAE (total, L*, ab*)
        """
        residuals = torch.abs(target - mu)
        norm_res = residuals / sigma

        huber = torch.where(
            norm_res < self.delta,
            0.5 * (norm_res ** 2),
            self.delta * (norm_res - 0.5 * self.delta),
        )
        nll_loss = (huber + 0.5 * log_var).mean()

        with torch.no_grad():
            mae_total = residuals.mean().item()
            mae_L = residuals[..., 0].mean().item()
            mae_ab = residuals[..., 1:].mean().item()

        metrics = {
            "loss": nll_loss.item(),
            "mae_total": mae_total,
            "mae_L": mae_L,
            "mae_ab": mae_ab,
        }
        return nll_loss, metrics
