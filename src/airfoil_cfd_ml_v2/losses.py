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


class GradientDifferenceLoss(nn.Module):
    def __init__(self, channel_weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.channel_weights = torch.tensor(channel_weights).view(1, -1, 1, 1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        spatial_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        err_dx = (pred_dx - tgt_dx).abs()
        err_dy = (pred_dy - tgt_dy).abs()

        weights = self.channel_weights.to(pred.device)
        err_dx = err_dx * weights
        err_dy = err_dy * weights

        if spatial_weight is None:
            return 0.5 * (err_dx.mean() + err_dy.mean())

        spatial_weight = spatial_weight.to(pred.device)
        weight_dx = spatial_weight[:, :, :, 1:] * spatial_weight[:, :, :, :-1]
        weight_dy = spatial_weight[:, :, 1:, :] * spatial_weight[:, :, :-1, :]

        weighted_dx = err_dx * weight_dx
        weighted_dy = err_dy * weight_dy

        denom_dx = (weight_dx.sum() * pred.shape[1]).clamp_min(1e-6)
        denom_dy = (weight_dy.sum() * pred.shape[1]).clamp_min(1e-6)
        return 0.5 * (weighted_dx.sum() / denom_dx + weighted_dy.sum() / denom_dy)


class CompositeV2Loss(nn.Module):
    def __init__(
        self,
        value_weights=(5.0, 1.0, 1.0),
        gradient_weights=(1.0, 1.0, 1.0),
        grad_loss_weight: float = 0.2,
        divergence_loss_weight: float = 0.0,
        divergence_dvdy_scale: float = 1.0,
        divergence_interior_margin: int = 0,
        divergence_use_physical_space: bool = False,
    ):
        super().__init__()
        self.value_loss = WeightedMultiChannelLoss(weights=value_weights)
        self.gradient_loss = GradientDifferenceLoss(channel_weights=gradient_weights)
        self.grad_loss_weight = float(max(0.0, grad_loss_weight))
        self.divergence_loss_weight = float(max(0.0, divergence_loss_weight))
        self.divergence_dvdy_scale = float(max(0.0, divergence_dvdy_scale))
        self.divergence_interior_margin = int(max(0, divergence_interior_margin))
        self.divergence_use_physical_space = bool(divergence_use_physical_space)

    @staticmethod
    def _denormalize_velocity(pred_norm: torch.Tensor, target_min: torch.Tensor, target_max: torch.Tensor) -> torch.Tensor:
        vel_norm = pred_norm[:, 1:3, :, :]
        vmin = target_min[1:3].to(pred_norm.device)
        vmax = target_max[1:3].to(pred_norm.device)
        return ((vel_norm + 1.0) / 2.0) * (vmax - vmin) + vmin

    @staticmethod
    def _divergence_loss(pred: torch.Tensor, spatial_weight: torch.Tensor | None = None) -> torch.Tensor:
        # velocity channels: [u, v] = pred[:,1], pred[:,2]
        u = pred[:, 1:2, :, :]
        v = pred[:, 2:3, :, :]

        du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]
        dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]
        # Legacy default uses du/dx + dv/dy, kept here for backward compatibility.
        div = du_dx[:, :, :-1, :] + dv_dy[:, :, :, :-1]
        abs_div = div.abs()

        if spatial_weight is None:
            return abs_div.mean()

        spatial_weight = spatial_weight.to(pred.device)
        weight = (
            spatial_weight[:, :, :-1, :-1]
            * spatial_weight[:, :, :-1, 1:]
            * spatial_weight[:, :, 1:, :-1]
            * spatial_weight[:, :, 1:, 1:]
        )
        weighted = abs_div * weight
        denom = weight.sum().clamp_min(1e-6)
        return weighted.sum() / denom

    def _divergence_loss_weighted(
        self,
        pred: torch.Tensor,
        spatial_weight: torch.Tensor | None = None,
        target_min: torch.Tensor | None = None,
        target_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.divergence_use_physical_space and target_min is not None and target_max is not None:
            vel = self._denormalize_velocity(pred, target_min=target_min, target_max=target_max)
            u = vel[:, 0:1, :, :]
            v = vel[:, 1:2, :, :]
        else:
            u = pred[:, 1:2, :, :]
            v = pred[:, 2:3, :, :]

        du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]
        dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]
        div = du_dx[:, :, :-1, :] + self.divergence_dvdy_scale * dv_dy[:, :, :, :-1]
        abs_div = div.abs()

        if spatial_weight is None:
            if self.divergence_interior_margin <= 0:
                return abs_div.mean()
            margin = self.divergence_interior_margin
            h, w = abs_div.shape[-2:]
            if h <= 2 * margin or w <= 2 * margin:
                return abs_div.mean()
            core = abs_div[:, :, margin : h - margin, margin : w - margin]
            return core.mean()

        spatial_weight = spatial_weight.to(pred.device)
        weight = (
            spatial_weight[:, :, :-1, :-1]
            * spatial_weight[:, :, :-1, 1:]
            * spatial_weight[:, :, 1:, :-1]
            * spatial_weight[:, :, 1:, 1:]
        )

        if self.divergence_interior_margin > 0:
            margin = self.divergence_interior_margin
            h, w = weight.shape[-2:]
            if h > 2 * margin and w > 2 * margin:
                inner = torch.zeros_like(weight)
                inner[:, :, margin : h - margin, margin : w - margin] = 1.0
                weight = weight * inner

        weighted = abs_div * weight
        denom = weight.sum().clamp_min(1e-6)
        return weighted.sum() / denom

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        spatial_weight: torch.Tensor | None = None,
        target_min: torch.Tensor | None = None,
        target_max: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_value = self.value_loss(pred, target, spatial_weight=spatial_weight)
        total_loss = loss_value

        if self.grad_loss_weight > 0.0:
            loss_grad = self.gradient_loss(pred, target, spatial_weight=spatial_weight)
            total_loss = total_loss + self.grad_loss_weight * loss_grad

        if self.divergence_loss_weight > 0.0:
            loss_div = self._divergence_loss_weighted(
                pred,
                spatial_weight=spatial_weight,
                target_min=target_min,
                target_max=target_max,
            )
            total_loss = total_loss + self.divergence_loss_weight * loss_div

        return total_loss
