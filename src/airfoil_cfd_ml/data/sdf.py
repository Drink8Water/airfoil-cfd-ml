"""Signed distance field and boundary distance computation from binary masks.

Uses scipy.ndimage.distance_transform_edt for Euclidean distance transforms.
No OpenCV dependency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def mask_to_sdf(
    mask: np.ndarray,
    truncate: float | None = 16.0,
) -> np.ndarray:
    """Compute signed distance field from a binary fluid mask.

    Convention:
      - mask == 1  → fluid region
      - mask == 0  → solid/obstacle region
      - SDF > 0 in fluid, SDF < 0 in solid, SDF = 0 at boundary.

    Uses two-pass Euclidean distance transform (scipy).

    Args:
        mask: (H, W) binary array, 1=fluid, 0=solid.
        truncate: If positive, clamp SDF to [-truncate, truncate].
                  If None, no truncation.

    Returns:
        (H, W) float32 array of signed distances.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        # All solid: no fluid region
        raise ValueError("mask has no fluid region (all zeros)")
    if mask_bool.all():
        # All fluid: no solid region
        raise ValueError("mask has no solid region (all ones)")

    fluid = mask_bool  # 1  in fluid
    solid = ~fluid  # 1  in solid

    # distance_transform_edt: for each foreground pixel (1), distance to
    # nearest background pixel (0). Background pixels get distance 0.
    #   edt(fluid):  fluid=1(fg) → d_to_solid in fluid;  0 in solid
    #   edt(solid):  solid=1(fg) → d_to_fluid in solid;  0 in fluid
    dist_in_fluid_to_solid = distance_transform_edt(
        fluid.astype(np.uint8)
    ).astype(np.float32)
    dist_in_solid_to_fluid = distance_transform_edt(
        solid.astype(np.uint8)
    ).astype(np.float32)

    # fluid region:  positive (dist to solid boundary)
    # solid region:  negative (dist to fluid boundary, negated)
    sdf = dist_in_fluid_to_solid - dist_in_solid_to_fluid

    if truncate is not None and truncate > 0:
        sdf = np.clip(sdf, -truncate, truncate)

    return sdf


def mask_to_boundary_distance(
    mask: np.ndarray,
    truncate: float | None = 16.0,
) -> np.ndarray:
    """Compute unsigned distance to the nearest wall for each fluid pixel.

    Solid pixels are set to 0. This is the positive part of the SDF.

    Args:
        mask: (H, W) binary array, 1=fluid, 0=solid.
        truncate: If positive, clamp distance to [0, truncate].
                  If None, no truncation.

    Returns:
        (H, W) float32 array of boundary distances (0 in solid).
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    fluid = mask.astype(bool)
    if not fluid.any():
        raise ValueError("mask has no fluid region (all zeros)")
    if fluid.all():
        raise ValueError("mask has no solid region (all ones)")

    # edt(fluid): fluid=1(fg) → distance from each fluid pixel to
    # nearest solid (background) pixel.  0 in solid.
    dist = distance_transform_edt(fluid.astype(np.uint8)).astype(np.float32)

    # Already 0 in solid, >0 in fluid — no need to mask.

    if truncate is not None and truncate > 0:
        dist = np.clip(dist, 0.0, truncate)

    return dist


def compute_mask_statistics(
    mask: np.ndarray,
) -> dict:
    """Compute summary statistics for a binary mask.

    Args:
        mask: (H, W) binary array, 1=fluid, 0=solid.

    Returns:
        Dict with keys: fluid_fraction, has_connectivity_issues hint.
    """
    fluid = mask.astype(bool)
    total = mask.size
    n_fluid = int(fluid.sum())
    n_solid = total - n_fluid
    fluid_fraction = n_fluid / total if total > 0 else 0.0

    # Quick connectivity check: are there isolated fluid pockets?
    # If the mask has holes in the solid region, distance_transform
    # handles them correctly, but we flag it for diagnostics.
    has_isolated_fluid = False
    if 0 < fluid_fraction < 1.0 and n_fluid > 0:
        from scipy.ndimage import label as nd_label

        labeled, n_labels = nd_label(mask.astype(np.int32))
        # Count labels that contain at least 1 pixel
        unique, counts = np.unique(labeled[mask.astype(bool)], return_counts=True)
        has_isolated_fluid = len(unique) > 1

    return {
        "total_pixels": total,
        "fluid_pixels": n_fluid,
        "solid_pixels": n_solid,
        "fluid_fraction": round(fluid_fraction, 6),
        "has_isolated_fluid": has_isolated_fluid,
    }
