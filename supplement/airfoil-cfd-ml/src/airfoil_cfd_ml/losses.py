from __future__ import annotations

import torch
import torch.nn as nn


class WeightedMultiChannelLoss(nn.Module):
    def __init__(self, weights=(5.0, 1.0, 1.0)):
        super().__init__()
        self.weights = torch.tensor(weights).view(1, -1, 1, 1)
        self.base_loss = nn.L1Loss(reduction="none")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        spatial_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        channel_losses = self.base_loss(pred, target)
        weights = self.weights.to(channel_losses.device)
        weighted_losses = channel_losses * weights

        if mask is None and spatial_weight is None:
            return weighted_losses.mean()

        effective_weight = torch.ones_like(weighted_losses[:, :1, :, :])

        if mask is not None:
            mask = mask.to(weighted_losses.device)
            if mask.dim() != 4 or mask.shape[1] != 1:
                raise ValueError("mask must have shape [batch, 1, height, width]")
            effective_weight = effective_weight * mask

        if spatial_weight is not None:
            spatial_weight = spatial_weight.to(weighted_losses.device)
            if spatial_weight.dim() != 4 or spatial_weight.shape[1] != 1:
                raise ValueError("spatial_weight must have shape [batch, 1, height, width]")
            effective_weight = effective_weight * spatial_weight

        weighted_masked_losses = weighted_losses * effective_weight
        denom = (effective_weight.sum() * weighted_losses.shape[1]).clamp_min(1e-6)
        return weighted_masked_losses.sum() / denom
