"""Evaluator: compute regression, physics, spectral, and efficiency metrics.

Produces:
  - eval_metrics.json   — aggregate metrics dict.
  - per_sample_metrics.csv — per-sample values.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from ..metrics.regression import compute_regression_metrics
from ..metrics.physics import (
    boundary_rel_mae,
    divergence_error,
    vorticity_error,
    wake_rel_mae,
)
from ..metrics.spectral import energy_spectrum_error, spectral_error
from ..metrics.efficiency import benchmark_latency, count_parameters
from ..utils.device import resolve_device


class Evaluator:
    """Run evaluation on a test DataLoader and write reports.

    Usage::

        evaluator = Evaluator(model)
        metrics = evaluator.evaluate(test_loader, output_dir="outputs/eval")
        # → eval_metrics.json + per_sample_metrics.csv
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
    ):
        self.device = device or resolve_device()
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        test_loader,
        output_dir: str | None = None,
    ) -> Dict[str, float]:
        """Compute all metrics and write reports.

        Args:
            test_loader: DataLoader yielding DataSample batches.
            output_dir: If given, write ``eval_metrics.json`` and
                ``per_sample_metrics.csv``.

        Returns:
            Aggregate metrics dict including regression, physics, spectral,
            and efficiency keys.
        """
        # Accumulators for batch-level metrics
        n_batches = 0
        accum_reg: Dict[str, float] = {}
        accum_div: float = 0.0
        accum_vort: float = 0.0
        accum_boundary: float = 0.0
        accum_wake: float = 0.0
        accum_spectral: float = 0.0
        accum_energy_spec: float = 0.0

        per_sample_rows: List[Dict[str, object]] = []

        for batch in test_loader:
            x = batch.x.to(self.device)
            y = batch.y.to(self.device)
            mask = batch.mask.to(self.device)

            pred = self.model(x)

            B = x.shape[0]

            # --- Per-sample regression + physics ---
            for i in range(B):
                p_i = pred[i:i+1]
                y_i = y[i:i+1]
                m_i = mask[i:i+1]

                row: Dict[str, object] = _extract_meta(batch.meta, i)

                # Regression
                reg = compute_regression_metrics(p_i, y_i, m_i)
                for k, v in reg.items():
                    row[k] = float(v)
                row["mean_rel_mae"] = (
                    float(reg.get("pressure_rel_mae", 0))
                    + float(reg.get("u_rel_mae", 0))
                    + float(reg.get("v_rel_mae", 0))
                ) / 3.0

                # Physics (per-sample)
                row["divergence_error"] = divergence_error(p_i, m_i)
                row["vorticity_error"] = vorticity_error(p_i, y_i, m_i)
                row["boundary_rel_mae"] = boundary_rel_mae(p_i, y_i, m_i)
                try:
                    row["wake_rel_mae"] = wake_rel_mae(p_i, y_i, m_i)
                except Exception:
                    row["wake_rel_mae"] = -1.0

                # Spectral (per-sample, for detailed analysis)
                row["spectral_error"] = spectral_error(p_i, y_i, m_i)
                row["energy_spectrum_error"] = energy_spectrum_error(p_i, y_i, m_i)

                per_sample_rows.append(row)

            # --- Batch-level accumulators ---
            m_batch = compute_regression_metrics(pred, y, mask)
            for k, v in m_batch.items():
                accum_reg[k] = accum_reg.get(k, 0.0) + v
            accum_div += divergence_error(pred, mask)
            accum_vort += vorticity_error(pred, y, mask)
            accum_boundary += boundary_rel_mae(pred, y, mask)
            try:
                accum_wake += wake_rel_mae(pred, y, mask)
            except Exception:
                pass
            accum_spectral += spectral_error(pred, y, mask)
            accum_energy_spec += energy_spectrum_error(pred, y, mask)

            n_batches += 1

        # --- Aggregate result ---
        nb = max(n_batches, 1)
        result: Dict[str, float] = {}
        for k, v in accum_reg.items():
            result[k] = v / nb
        result["mean_rel_mae"] = (
            result.get("pressure_rel_mae", 0.0)
            + result.get("u_rel_mae", 0.0)
            + result.get("v_rel_mae", 0.0)
        ) / 3.0
        # Physics
        result["divergence_error"] = accum_div / nb
        result["vorticity_error"] = accum_vort / nb
        result["boundary_rel_mae"] = accum_boundary / nb if n_batches > 0 else -1.0
        result["wake_rel_mae"] = accum_wake / max(nb, 1)
        # Spectral
        result["spectral_error"] = accum_spectral / nb
        result["energy_spectrum_error"] = accum_energy_spec / nb
        # Efficiency
        result["n_parameters"] = float(count_parameters(self.model, trainable_only=True))
        try:
            in_c = _guess_in_channels(self.model)
            lat = benchmark_latency(
                self.model,
                input_shape=(1, in_c, 128, 128),
                device=self.device,
                warmup=3,
                repeat=10,
            )
            result["latency_ms_mean"] = lat["mean_ms"]
            result["latency_ms_std"] = lat["std_ms"]
            result["device"] = lat["device"]
            result["device_name"] = lat["device_name"]
            result["input_shape"] = lat["input_shape"]
            result["latency_warmup"] = lat["warmup"]
            result["latency_repeat"] = lat["repeat"]
        except Exception:
            result["latency_ms_mean"] = -1.0
            result["latency_ms_std"] = -1.0
            result["device"] = str(self.device)
            result["device_name"] = str(self.device)
            result["input_shape"] = [-1, -1, -1, -1]
            result["latency_warmup"] = -1
            result["latency_repeat"] = -1

        # --- Write reports ---
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            (out / "eval_metrics.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if per_sample_rows:
                csv_path = out / "per_sample_metrics.csv"
                fieldnames = list(per_sample_rows[0].keys())
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(per_sample_rows)

        return result


def _guess_in_channels(model: nn.Module) -> int:
    """Guess the input channel count from the first Conv2d in the model."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    return 3  # fallback


def _extract_meta(meta, idx: int) -> Dict[str, object]:
    """Safely extract metadata for sample ``idx`` from a batch meta dict."""
    row: Dict[str, object] = {}
    if not isinstance(meta, dict):
        return row
    for key in ("file_name", "index", "geometry_mode"):
        val = meta.get(key)
        if isinstance(val, (list, tuple)):
            row[key] = str(val[idx]) if idx < len(val) else ""
        elif val is not None:
            row[key] = str(val)
    return row
