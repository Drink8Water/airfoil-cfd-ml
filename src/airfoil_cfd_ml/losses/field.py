"""FieldLoss: per-pixel MSE or MAE with optional channel weights and fluid mask.

Returns (total_loss, loss_dict) — unified interface for all losses.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class FieldLoss(nn.Module):
    """Per-pixel regression loss with channel weights and optional spatial mask.

    Args:
        reduction: "mse" or "mae".
        channel_weights: weights for [pressure, u, v] channels.

    Returns:
        (total_loss: scalar Tensor, loss_dict: Dict[str, float])
    """

    def __init__(
        self,
        reduction: str = "mse",
        channel_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        if reduction not in ("mse", "mae"):
            raise ValueError(f"Unknown reduction '{reduction}'; use 'mse' or 'mae'.")
        self.reduction = reduction
        self.register_buffer(
            "channel_weights",
            torch.tensor(channel_weights, dtype=torch.float32).view(1, -1, 1, 1),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute field loss.

        Args:
            pred: (B, 3, H, W) prediction.
            target: (B, 3, H, W) ground truth.
            mask: (B, 1, H, W) optional fluid mask (1=fluid).

        Returns:
            (total_loss, loss_dict) where loss_dict has keys:
            total, loss_pressure, loss_u, loss_v.
        """
        if self.reduction == "mse":
            per_element = (pred - target).pow(2)
        else:
            per_element = (pred - target).abs()

        # Apply channel weights
        channel_losses = per_element * self.channel_weights.to(pred.device)

        if mask is not None:
            channel_losses = channel_losses * mask
            denom = (mask.sum() * per_element.shape[1]).clamp_min(1e-6)
            total = channel_losses.sum() / denom
        else:
            total = channel_losses.mean()

        # Per-channel losses (detached, for logging)
        ch_names = ["pressure", "u", "v"]
        loss_dict: Dict[str, float] = {"total": total.item()}
        with torch.no_grad():
            for i, name in enumerate(ch_names):
                if mask is not None:
                    ch_denom = mask.sum().clamp_min(1e-6)
                    ch_val = (
                        (per_element[:, i : i + 1] * mask).sum() / ch_denom
                    ).item()
                else:
                    ch_val = per_element[:, i : i + 1].mean().item()
                loss_dict[f"loss_{name}"] = ch_val

        return total, loss_dict
