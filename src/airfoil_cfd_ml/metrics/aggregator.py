"""Metric aggregation: combine per-sample metrics into global statistics."""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def aggregate_metric_dicts(
    per_sample_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate a list of per-sample metric dicts into mean/std statistics.

    Only numeric (float/int) values are aggregated.  Non-numeric values
    are silently dropped.

    Args:
        per_sample_metrics: List of dicts, each with the same keys.

    Returns:
        Dict with keys like ``{key}_mean``, ``{key}_std``, and
        ``n_samples``.
    """
    if not per_sample_metrics:
        return {"n_samples": 0}

    n = len(per_sample_metrics)

    # Collect all numeric keys
    keys: List[str] = []
    for k, v in per_sample_metrics[0].items():
        if isinstance(v, (int, float)):
            keys.append(k)

    result: Dict[str, Any] = {"n_samples": n}
    for key in keys:
        vals = []
        for row in per_sample_metrics:
            v = row.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            t = torch.tensor(vals, dtype=torch.float32)
            result[f"{key}_mean"] = round(float(t.mean().item()), 8)
            result[f"{key}_std"] = round(float(t.std().item()), 8)

    return result
