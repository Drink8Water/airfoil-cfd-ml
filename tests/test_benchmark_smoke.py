"""Smoke test for the benchmark runner: train + evaluate + CSV output.

Uses synthetic npz data via tempfile — no real data required.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CANONICAL_KEY = "a"


def _make_synthetic_npz_dir(base_dir: str, n_files: int, h: int, w: int) -> str:
    """Create a directory of synthetic .npz files."""
    d = os.path.join(base_dir, f"synth_{n_files}")
    os.makedirs(d, exist_ok=True)
    for i in range(n_files):
        arr = np.zeros((6, h, w), dtype=np.float32)
        arr[0] = np.float32(0.8)
        arr[1] = np.float32(0.0)
        yy, xx = np.ogrid[:h, :w]
        cx, cy, r = w // 2, h // 2, 8
        solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        arr[2] = solid.astype(np.float32)
        arr[3] = np.random.randn(h, w).astype(np.float32) * 0.1
        arr[4] = np.random.randn(h, w).astype(np.float32) * 5.0
        arr[5] = np.random.randn(h, w).astype(np.float32) * 5.0
        np.savez_compressed(os.path.join(d, f"sample_{i:04d}.npz"), **{CANONICAL_KEY: arr})
    return d


class TestBenchmarkSmoke:
    """End-to-end: 1-config benchmark run."""

    def test_single_config_train_eval_csv(self):
        """Train+eval one config via benchmark functions, verify CSV output."""
        tmpdir = tempfile.mkdtemp()
        train_dir = _make_synthetic_npz_dir(tmpdir, 6, h=64, w=64)
        val_dir = _make_synthetic_npz_dir(
            os.path.join(tmpdir, "val"), 4, h=64, w=64
        )

        # Write a minimal config
        config_dir = os.path.join(tmpdir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "test_bench.yaml")
        cfg = {
            "data": {
                "train_dir": train_dir,
                "val_dir": val_dir,
                "batch_size": 4,
                "num_workers": 0,
                "geometry_mode": "mask_only",
            },
            "model": {
                "name": "simple_cnn",
                "kwargs": {"hidden_channels": 8, "n_layers": 2, "out_channels": 3},
            },
            "loss": {
                "name": "field",
                "kwargs": {"reduction": "mse", "channel_weights": [1.0, 1.0, 1.0]},
            },
            "train": {
                "epochs": 1,
                "lr": 0.001,
                "weight_decay": 0.0,
                "seed": 42,
                "save_dir": f"{tmpdir}/outputs/test_bench",
            },
        }
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

        # Run via the benchmark module's internal functions
        from scripts.run_benchmark import _train_one, _evaluate_one

        # Train
        train_result = _train_one(config_path, prefer_cuda=False)
        assert os.path.isfile(train_result["best_pt"]), "best.pt not created"

        # Evaluate
        metrics = _evaluate_one(
            train_result["best_pt"],
            cfg["data"]["geometry_mode"],
            val_dir,
            os.path.join(train_result["save_dir"], "eval"),
            prefer_cuda=False,
        )

        # Check key metrics exist
        assert "mean_rel_mae" in metrics
        assert "divergence_error" in metrics
        assert "vorticity_error" in metrics
        assert "boundary_rel_mae" in metrics
        assert "spectral_error" in metrics
        assert "energy_spectrum_error" in metrics
        assert "n_parameters" in metrics
        assert "latency_ms_mean" in metrics

        # Check JSON + CSV exist
        eval_dir = os.path.join(train_result["save_dir"], "eval")
        assert os.path.isfile(os.path.join(eval_dir, "eval_metrics.json"))
        assert os.path.isfile(os.path.join(eval_dir, "per_sample_metrics.csv"))

        # Check per-sample CSV contains physics columns
        with open(os.path.join(eval_dir, "per_sample_metrics.csv"), "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == len(os.listdir(val_dir))
            first = rows[0]
            for key in ("divergence_error", "boundary_rel_mae", "spectral_error"):
                assert key in first, f"Missing column: {key}"

    def test_benchmark_runner_csv(self):
        """Full run_benchmark flow: 2 configs → CSV with expected columns."""
        tmpdir = tempfile.mkdtemp()
        train_dir = _make_synthetic_npz_dir(tmpdir, 6, h=64, w=64)
        val_dir = _make_synthetic_npz_dir(
            os.path.join(tmpdir, "val"), 4, h=64, w=64
        )

        config_dir = os.path.join(tmpdir, "configs")
        os.makedirs(config_dir, exist_ok=True)

        # Two configs
        for i, geo in enumerate(["mask_only", "mask_xy"]):
            cfg = {
                "data": {
                    "train_dir": train_dir,
                    "val_dir": val_dir,
                    "batch_size": 4,
                    "num_workers": 0,
                    "geometry_mode": geo,
                },
                "model": {
                    "name": "simple_cnn",
                    "kwargs": {"hidden_channels": 8, "n_layers": 2, "out_channels": 3},
                },
                "loss": {
                    "name": "field",
                    "kwargs": {"reduction": "mse"},
                },
                "train": {
                    "epochs": 1,
                    "lr": 0.001,
                    "seed": 42,
                    "save_dir": f"{tmpdir}/outputs/bench_{i}",
                },
            }
            with open(os.path.join(config_dir, f"bench_{i}.yaml"), "w") as f:
                yaml.dump(cfg, f)

        # Run
        csv_path = os.path.join(tmpdir, "main_benchmark_smoke.csv")
        from scripts.run_benchmark import _train_one, _evaluate_one

        for cf in sorted(Path(config_dir).glob("*.yaml")):
            tr = _train_one(str(cf), prefer_cuda=False)
            with open(cf) as f:
                c = yaml.safe_load(f)
            _evaluate_one(
                tr["best_pt"],
                c["data"]["geometry_mode"],
                val_dir,
                os.path.join(tr["save_dir"], "eval"),
                prefer_cuda=False,
            )

        # Build a minimal CSV manually (simulating the full run_benchmark output)
        from airfoil_cfd_ml.evaluation.leaderboard import collect_eval_results, save_leaderboard_csv

        rows = collect_eval_results(f"{tmpdir}/outputs")
        save_leaderboard_csv(rows, csv_path)

        assert os.path.isfile(csv_path)
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 2

        # Expected columns in benchmark CSV
        expected_cols = {"experiment", "mean_rel_mae", "divergence_error", "n_parameters"}
        found = set(csv_rows[0].keys())
        for col in expected_cols:
            assert col in found, f"Missing column: {col}"
