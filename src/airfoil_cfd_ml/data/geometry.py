"""Geometry encoding: coordinate grids and channel augmentation.

Provides:
  - build_coordinate_grid: normalized XY coordinate channels.
  - augment_geometry_channels: add geometry channels based on mode string.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .sdf import mask_to_boundary_distance, mask_to_sdf

# Canonical geometry modes
GEOMETRY_MODES = frozenset(
    {
        "mask_only",
        "mask_xy",
        "mask_xy_sdf",
        "mask_xy_sdf_boundary",
    }
)

# Map mode → expected input channel count (before augmentation)
MODE_CHANNEL_MAP: dict[str, int] = {
    "mask_only": 3,
    "mask_xy": 5,
    "mask_xy_sdf": 6,
    "mask_xy_sdf_boundary": 7,
}


def build_coordinate_grid(height: int, width: int) -> torch.Tensor:
    """Create normalized XY coordinate channels.

    Coordinates are in [-1, 1]:
      x: [-1            … +1  ]   left → right
      y: [-1 (top) … +1 (bottom)] (image convention: row 0 = top)

    Args:
        height: spatial height (rows).
        width: spatial width (cols).

    Returns:
        Tensor of shape (2, height, width), float32.
    """
    # Pixel centres
    yy = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    xx = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)

    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack([gx, gy], dim=0)  # (2, H, W)


def augment_geometry_channels(
    x: torch.Tensor,
    mode: str = "mask_only",
    sdf_truncate: float = 16.0,
) -> torch.Tensor:
    """Augment the standard 3-channel input with geometry channels.

    Args:
        x: Input tensor of shape (3, H, W) containing
           [u_inf_x, u_inf_y, mask].
        mode: One of 'mask_only', 'mask_xy', 'mask_xy_sdf',
              'mask_xy_sdf_boundary'.
        sdf_truncate: Truncation distance for SDF / boundary distance
                      (in pixels). Default 16 for 128×128 grids.

    Returns:
        Augmented tensor of shape (C_out, H, W) where C_out depends on mode:
          mask_only             → 3
          mask_xy               → 5
          mask_xy_sdf           → 6
          mask_xy_sdf_boundary  → 7

    Raises:
        ValueError: if mode is unknown or x has wrong shape.
    """
    if mode not in GEOMETRY_MODES:
        raise ValueError(
            f"Unknown geometry_mode '{mode}'. Choose from {sorted(GEOMETRY_MODES)}."
        )

    if x.ndim != 3:
        raise ValueError(f"x must be 3D (C,H,W), got shape {x.shape}")
    if x.shape[0] != 3:
        raise ValueError(
            f"Expected 3 input channels [u_inf_x, u_inf_y, mask], "
            f"got {x.shape[0]} channels."
        )

    if mode == "mask_only":
        return x

    H, W = x.shape[1], x.shape[2]
    channels: list[torch.Tensor] = [x]

    # XY coordinates
    if mode in ("mask_xy", "mask_xy_sdf", "mask_xy_sdf_boundary"):
        coords = build_coordinate_grid(H, W)
        channels.append(coords)

    # SDF and boundary distance require numpy (scipy)
    if mode in ("mask_xy_sdf", "mask_xy_sdf_boundary"):
        mask_np = x[2].cpu().numpy()  # (H, W)
        # 1 = fluid: mask < 0.5
        fluid_mask_np = (mask_np < 0.5).astype(np.uint8)

        sdf_np = mask_to_sdf(fluid_mask_np, truncate=sdf_truncate)
        # Normalise SDF to [-1, 1] range (0 at boundary)
        sdf_ch = torch.from_numpy(sdf_np).unsqueeze(0).float()
        if sdf_truncate > 0:
            sdf_ch = sdf_ch / sdf_truncate
        channels.append(sdf_ch)

    if mode == "mask_xy_sdf_boundary":
        mask_np = x[2].cpu().numpy()
        fluid_mask_np = (mask_np < 0.5).astype(np.uint8)

        bdist_np = mask_to_boundary_distance(fluid_mask_np, truncate=sdf_truncate)
        bdist_ch = torch.from_numpy(bdist_np).unsqueeze(0).float()
        if sdf_truncate > 0:
            bdist_ch = bdist_ch / sdf_truncate
        channels.append(bdist_ch)

    return torch.cat(channels, dim=0)


def get_input_channels(mode: str) -> int:
    """Return the total number of input channels for a given geometry mode.

    Args:
        mode: geometry mode string.

    Returns:
        Number of input channels.
    """
    if mode not in MODE_CHANNEL_MAP:
        raise ValueError(
            f"Unknown geometry_mode '{mode}'. Choose from {sorted(MODE_CHANNEL_MAP.keys())}."
        )
    return MODE_CHANNEL_MAP[mode]
