"""Regression metrics: MAE, RMSE, RelMAE per channel (pressure, u, v).

All functions return Dict[str, float].
"""

from __future__ import annotations

from typing import Dict

import torch


def compute_regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Compute per-channel MAE, RMSE, and Relative MAE.

    Args:
        pred: (B, 3, H, W) prediction.
        target: (B, 3, H, W) ground truth.
        mask: (B, 1, H, W) optional fluid mask.

    Returns:
        Dict with keys: {ch}_mae, {ch}_rmse, {ch}_rel_mae for ch in [pressure, u, v].
    """
    ch_names = ["pressure", "u", "v"]

    if mask is not None:
        pred = pred * mask
        target = target * mask
        n = mask.sum().item()
    else:
        n = pred[0, 0].numel()

    n = max(n, 1)

    metrics: Dict[str, float] = {}
    for i, name in enumerate(ch_names):
        err = pred[:, i : i + 1] - target[:, i : i + 1]
        abs_err = err.abs()
        mae = abs_err.sum().item() / n
        rmse = (err.pow(2).sum().item() / n) ** 0.5
        tgt_abs = target[:, i : i + 1].abs().sum().item()
        rel_mae = abs_err.sum().item() / max(tgt_abs, 1e-8)
        metrics[f"{name}_mae"] = float(mae)
        metrics[f"{name}_rmse"] = float(rmse)
        metrics[f"{name}_rel_mae"] = float(rel_mae)

    return metrics
