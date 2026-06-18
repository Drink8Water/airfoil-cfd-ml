"""Unified data schema for the SciML framework.

All datasets MUST return a DataSample with these fields:
  x    — input tensor  (C, H, W)
  y    — target tensor (3, H, W)  [pressure, u_flow, v_flow]
  mask — fluid mask     (1, H, W)  [1=fluid, 0=solid/obstacle]
  meta — dict of metadata (file_name, index, etc.)

Validation:
  validate_npz_file(path) checks key, shape, dtype, NaN/Inf, and mask binarity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Canonical npz key for airfoil data.
CANONICAL_KEY = "a"
# Expected spatial resolution (H, W). If None, any square-ish size is accepted.
EXPECTED_SPATIAL = (128, 128)


@dataclass
class DataSample:
    """Single sample with mandatory x, y, mask, meta fields."""

    x: torch.Tensor  # (C, H, W)
    y: torch.Tensor  # (3, H, W)
    mask: torch.Tensor  # (1, H, W)
    meta: Dict[str, Any] = field(default_factory=dict)


def collate_stack(samples: List[DataSample]) -> DataSample:
    """Default collate: stack a list of DataSample into a batched DataSample.

    Args:
        samples: list of DataSample, each with same-shape tensors.

    Returns:
        Batched DataSample with tensors stacked along dim=0.
    """
    return DataSample(
        x=torch.stack([s.x for s in samples]),
        y=torch.stack([s.y for s in samples]),
        mask=torch.stack([s.mask for s in samples]),
        meta={"n_samples": len(samples)},
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_MASK_BINARITY_TOLERANCE = 0.05  # fraction of non-binary pixels tolerated


def validate_npz_file(
    path: str,
    expected_spatial: Optional[Tuple[int, int]] = EXPECTED_SPATIAL,
) -> Dict[str, Any]:
    """Validate a single .npz airfoil sample file.

    Checks performed:
      1. File exists and is readable.
      2. Key ``'a'`` is present.
      3. Array shape is ``(6, H, W)``.
      4. Dtype is float32 (or castable without loss).
      5. No NaN or Inf values.
      6. Mask channel (index 2) is approximately binary.

    Args:
        path: Path to a .npz file.
        expected_spatial: Expected (H, W) spatial shape, or None to skip
            the spatial check.

    Returns:
        Dict with keys:
          valid (bool), path (str), errors (list[str]), warnings (list[str]),
          shape (tuple), dtype (str), mask_stats (dict or None).

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    result: Dict[str, Any] = {
        "valid": True,
        "path": os.path.abspath(path),
        "errors": [],
        "warnings": [],
        "shape": None,
        "dtype": None,
        "mask_stats": None,
    }

    # 1. Existence
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    # 2. Load and check key
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        result["valid"] = False
        result["errors"].append(f"Cannot load npz: {exc}")
        return result

    if CANONICAL_KEY not in data:
        result["valid"] = False
        result["errors"].append(
            f"Missing key '{CANONICAL_KEY}'. Found keys: {list(data.keys())}"
        )
        # Don't return yet — try to give as much info as possible.
        return result

    arr = data[CANONICAL_KEY]
    result["shape"] = tuple(arr.shape)
    result["dtype"] = str(arr.dtype)

    # 3. Shape: (6, H, W)
    if arr.ndim != 3:
        result["valid"] = False
        result["errors"].append(
            f"Expected 3D array (6,H,W), got {arr.ndim}D shape {arr.shape}"
        )
        return result

    C, H, W = arr.shape
    if C != 6:
        result["valid"] = False
        result["errors"].append(
            f"Expected 6 channels, got {C} (shape {arr.shape})"
        )

    if expected_spatial is not None:
        eh, ew = expected_spatial
        if H != eh or W != ew:
            result["warnings"].append(
                f"Spatial shape ({H},{W}) differs from expected ({eh},{ew})"
            )

    # 4. Dtype – warn if not float32
    if arr.dtype != np.float32:
        result["warnings"].append(
            f"Array dtype is {arr.dtype}, expected float32"
        )

    # 5. NaN / Inf
    if np.any(~np.isfinite(arr)):
        result["valid"] = False
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        result["errors"].append(
            f"Array contains non-finite values: {nan_count} NaN, {inf_count} Inf"
        )

    # 6. Mask binarity (channel 2)
    if C >= 3:
        mask_ch = arr[2]
        unique_vals = np.unique(mask_ch)
        # Heuristic: mask should be mostly 0 or 1
        # Count how many pixels differ from 0 or 1 by more than 1e-3
        not_binary = (
            (np.abs(mask_ch - 0.0) > 1e-3) & (np.abs(mask_ch - 1.0) > 1e-3)
        )
        n_not_binary = int(np.sum(not_binary))
        frac_not_binary = n_not_binary / mask_ch.size

        mask_stats = {
            "unique_values": [round(float(v), 4) for v in unique_vals],
            "n_not_binary": n_not_binary,
            "fraction_not_binary": round(float(frac_not_binary), 6),
            "is_binary": frac_not_binary < _MASK_BINARITY_TOLERANCE,
        }
        result["mask_stats"] = mask_stats

        if not mask_stats["is_binary"]:
            result["warnings"].append(
                f"Mask channel is not approximately binary "
                f"({frac_not_binary:.4f} of pixels differ from {{0,1}} by >1e-3)"
            )

    return result
