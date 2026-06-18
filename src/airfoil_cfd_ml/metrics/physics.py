"""Physics-aware evaluation metrics for 2D airfoil flow fields.

Conventions:
  - pred / target: (B, 3, H, W) — channels: 0=pressure, 1=u, 2=v.
  - mask: (B, 1, H, W) or (B, H, W) — 1=fluid, 0=solid.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Finite-difference operators
# ---------------------------------------------------------------------------

def finite_diff_x(u: torch.Tensor) -> torch.Tensor:
    """Central-difference approximation of ∂u/∂x.

    Uses second-order central difference on interior columns and
    first-order one-sided differences on the boundary columns.

    Args:
        u: Tensor of shape (B, C, H, W) or (B, H, W).

    Returns:
        Tensor of same shape as u.
    """
    du = torch.zeros_like(u)
    # Interior: (u_{i+1} - u_{i-1}) / 2
    du[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / 2.0
    # Left edge: u_1 - u_0
    du[..., 0] = u[..., 1] - u[..., 0]
    # Right edge: u_{-1} - u_{-2}
    du[..., -1] = u[..., -1] - u[..., -2]
    return du


def finite_diff_y(u: torch.Tensor) -> torch.Tensor:
    """Central-difference approximation of ∂u/∂y.

    Args:
        u: Tensor of shape (B, C, H, W) or (B, H, W).

    Returns:
        Tensor of same shape as u.
    """
    du = torch.zeros_like(u)
    # Interior: (u_{j+1} - u_{j-1}) / 2
    du[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / 2.0
    # Top edge: u_1 - u_0
    du[..., 0, :] = u[..., 1, :] - u[..., 0, :]
    # Bottom edge: u_{-1} - u_{-2}
    du[..., -1, :] = u[..., -1, :] - u[..., -2, :]
    return du


# ---------------------------------------------------------------------------
# Physics metrics
# ---------------------------------------------------------------------------

def _ensure_mask_4d(mask: Optional[torch.Tensor], like: torch.Tensor) -> Optional[torch.Tensor]:
    """Ensure mask is (B,1,H,W)."""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask


def divergence_error(
    pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Mean absolute divergence |∂u/∂x + ∂v/∂y| of the predicted velocity field.

    For incompressible flow this should be 0.

    Args:
        pred: (B, 3, H, W) — channels: 0=pressure, 1=u, 2=v.
        mask: Optional (B, 1, H, W) fluid mask.

    Returns:
        Scalar float: mean absolute divergence over fluid pixels.
    """
    u = pred[:, 1:2, :, :]  # (B, 1, H, W)
    v = pred[:, 2:3, :, :]  # (B, 1, H, W)

    du_dx = finite_diff_x(u)
    dv_dy = finite_diff_y(v)

    div = du_dx + dv_dy
    abs_div = div.abs()

    mask = _ensure_mask_4d(mask, pred)
    if mask is not None:
        n = mask.sum().clamp_min(1)
        return float((abs_div * mask).sum().item() / n.item())

    return float(abs_div.mean().item())


def vorticity_error(
    pred: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Mean absolute error of vorticity ω = ∂v/∂x − ∂u/∂y.

    If target is provided, computes MAE between ω_pred and ω_target.
    If target is None, computes mean |ω_pred| (reference: physical vorticity
    for irrotational flow should be near zero outside boundary layers).

    Args:
        pred: (B, 3, H, W).
        target: Optional (B, 3, H, W).
        mask: Optional fluid mask.

    Returns:
        Scalar float: mean absolute vorticity error.
    """
    u_p = pred[:, 1:2]
    v_p = pred[:, 2:3]
    omega_pred = finite_diff_x(v_p) - finite_diff_y(u_p)

    if target is not None:
        u_t = target[:, 1:2]
        v_t = target[:, 2:3]
        omega_target = finite_diff_x(v_t) - finite_diff_y(u_t)
        err = (omega_pred - omega_target).abs()
    else:
        err = omega_pred.abs()

    mask = _ensure_mask_4d(mask, pred)
    if mask is not None:
        n = mask.sum().clamp_min(1)
        return float((err * mask).sum().item() / n.item())

    return float(err.mean().item())


def boundary_rel_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    ring_width: int = 3,
) -> float:
    """Mean relative MAE restricted to a narrow ring around the fluid/solid boundary.

    The boundary ring is extracted by morphological dilation minus erosion
    (using max-pool approximation), then intersected with the fluid mask.

    Args:
        pred: (B, 3, H, W).
        target: (B, 3, H, W).
        mask: (B, 1, H, W) or (B, H, W) — 1=fluid, 0=solid.
        ring_width: Width of the boundary ring in pixels (default 3).

    Returns:
        mean_rel_mae computed over boundary-ring fluid pixels only.
    """
    mask = _ensure_mask_4d(mask, pred)  # (B, 1, H, W)
    if mask is None:
        raise ValueError("mask is required for boundary_rel_mae")

    # solid = 1 - mask (1 where solid)
    solid = 1.0 - mask

    kernel = 2 * ring_width + 1
    pad = ring_width

    # Dilation of solid: max_pool
    solid_dilated = -F.max_pool2d(
        -solid, kernel_size=kernel, stride=1, padding=pad
    )
    # Erosion of solid: -max_pool(-solid) already did dilation; erosion = min_pool
    # Use: erosion = -max_pool(-solid) with large kernel? No.
    # Simpler: dilate solid, dilate fluid, intersect.
    fluid_dilated = -F.max_pool2d(
        -(1.0 - solid), kernel_size=kernel, stride=1, padding=pad
    )

    # Boundary ring = (dilated_solid AND fluid) region
    boundary = (solid_dilated * mask).clamp(0.0, 1.0)

    n_boundary = boundary.sum()
    if n_boundary < 1:
        return 0.0

    abs_err = (pred - target).abs()
    tgt_abs = target.abs()

    # Channel-wise rel_mae over boundary only
    rel_sum = 0.0
    for c in range(3):
        ch_abs = (abs_err[:, c:c+1] * boundary).sum()
        ch_tgt = (tgt_abs[:, c:c+1] * boundary).sum().clamp_min(1e-8)
        rel_sum += (ch_abs / ch_tgt).item()

    return float(rel_sum / 3.0)


def wake_rel_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    wake_ratio: float = 0.3,
) -> float:
    """Mean relative MAE restricted to the wake region (right of the airfoil).

    The wake region is defined as fluid pixels whose x-coordinate is greater
    than the rightmost solid pixel for that sample.  Additionally, a margin
    proportional to ``wake_ratio`` of the domain width is added.

    Args:
        pred: (B, 3, H, W).
        target: (B, 3, H, W).
        mask: (B, 1, H, W) or (B, H, W) — 1=fluid, 0=solid.
        wake_ratio: Fraction of domain width to extend the wake region
            beyond the airfoil trailing edge (default 0.3).  A value of 1.0
            means the entire region right of the trailing edge.

    Returns:
        mean_rel_mae computed over wake-region pixels.
    """
    mask = _ensure_mask_4d(mask, pred)
    if mask is None:
        raise ValueError("mask is required for wake_rel_mae")

    B, _, H, W = pred.shape
    solid = (1.0 - mask).squeeze(1)  # (B, H, W)

    wake_masks = torch.zeros_like(mask)

    for b in range(B):
        # Find rightmost solid pixel
        solid_b = solid[b]  # (H, W)
        cols_with_solid = solid_b.sum(dim=0) > 0  # (W,)
        if cols_with_solid.any():
            rightmost = int(cols_with_solid.nonzero(as_tuple=True)[0][-1].item())
        else:
            rightmost = W // 2  # fallback: use mid-domain

        # Wake starts at rightmost solid column
        wake_start = rightmost + 1
        wake_width = max(int(W * wake_ratio), 1)
        wake_end = min(wake_start + wake_width, W)

        if wake_end > wake_start:
            wake_masks[b, 0, :, wake_start:wake_end] = 1.0

    # Intersect with fluid mask
    wake_region = wake_masks * mask
    n_wake = wake_region.sum()
    if n_wake < 1:
        return 0.0

    abs_err = (pred - target).abs()
    tgt_abs = target.abs()

    rel_sum = 0.0
    for c in range(3):
        ch_abs = (abs_err[:, c:c+1] * wake_region).sum()
        ch_tgt = (tgt_abs[:, c:c+1] * wake_region).sum().clamp_min(1e-8)
        rel_sum += (ch_abs / ch_tgt).item()

    return float(rel_sum / 3.0)
