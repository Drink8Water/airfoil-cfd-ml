"""Dedicated tests for regression metrics: MAE, RMSE, RelMAE.

Uses synthetic tensors — no real data required.
"""

from __future__ import annotations

import pytest
import torch

from airfoil_cfd_ml.metrics.regression import compute_regression_metrics


# ======================================================================
# Helpers
# ======================================================================

def _make_tensors(b: int = 2, c: int = 3, h: int = 64, w: int = 64):
    pred = torch.randn(b, c, h, w)
    target = torch.randn(b, c, h, w)
    mask = torch.ones(b, 1, h, w)
    return pred, target, mask


# ======================================================================
# Expected keys
# ======================================================================

class TestMetricsKeys:
    def test_all_9_keys_present(self):
        pred, target, mask = _make_tensors()
        m = compute_regression_metrics(pred, target, mask)
        expected = {
            "pressure_mae", "pressure_rmse", "pressure_rel_mae",
            "u_mae", "u_rmse", "u_rel_mae",
            "v_mae", "v_rmse", "v_rel_mae",
        }
        assert set(m.keys()) == expected

    def test_all_values_nonnegative(self):
        pred, target, mask = _make_tensors()
        m = compute_regression_metrics(pred, target, mask)
        for v in m.values():
            assert v >= 0.0

    def test_all_values_float(self):
        pred, target, mask = _make_tensors()
        m = compute_regression_metrics(pred, target, mask)
        for v in m.values():
            assert isinstance(v, float)


# ======================================================================
# Zero-error case
# ======================================================================

class TestMetricsZeroError:
    def test_identical_pred_target(self):
        pred = torch.ones(2, 3, 16, 16)
        target = pred.clone()
        mask = torch.ones(2, 1, 16, 16)
        m = compute_regression_metrics(pred, target, mask)
        for key in m:
            assert m[key] == pytest.approx(0.0, abs=1e-5), f"{key} not zero: {m[key]}"

    def test_rel_mae_zero_for_identical(self):
        pred = torch.ones(1, 3, 32, 32)
        target = pred.clone()
        mask = torch.ones(1, 1, 32, 32)
        m = compute_regression_metrics(pred, target, mask)
        assert m["pressure_rel_mae"] == pytest.approx(0.0, abs=1e-5)


# ======================================================================
# Mask
# ======================================================================

class TestMetricsMask:
    def test_full_solid_zero_metrics(self):
        """When mask=0 everywhere, metrics should be 0 (no fluid)."""
        pred = torch.randn(1, 3, 8, 8)
        target = torch.randn(1, 3, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        m = compute_regression_metrics(pred, target, mask)
        # With mask=0, the denominator n = max(mask.sum(), 1) = 1, so values
        # may be near 0 since pred*mask = 0 and target*mask = 0.
        for v in m.values():
            assert 0.0 <= v < 1e-3

    def test_partial_mask(self):
        pred = torch.zeros(1, 3, 4, 4)
        target = torch.ones(1, 3, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        mask[:, :, 2:, :] = 0.0
        m = compute_regression_metrics(pred, target, mask)
        # Only top half (2×4=8 pixels) contribute
        # MAE should be ~1.0
        assert 0.5 < m["pressure_mae"] < 2.0


# ======================================================================
# RelMAE sanity
# ======================================================================

class TestRelMAE:
    def test_rel_mae_defined(self):
        """RelMAE should be well-defined when target has non-trivial magnitude."""
        pred = torch.zeros(1, 3, 16, 16)
        target = torch.ones(1, 3, 16, 16) * 10.0
        mask = torch.ones(1, 1, 16, 16)
        m = compute_regression_metrics(pred, target, mask)
        # MAE = 10, sum(|target|) = 10 * 256 = 2560, rel_mae = 2560 / 2560 = 1.0
        # Actually: abs_err.sum() = 10*256 = 2560, tgt_abs.sum() = 10*256 = 2560
        # rel_mae = 2560 / 2560 = 1.0
        for ch in ("pressure", "u", "v"):
            assert m[f"{ch}_rel_mae"] == pytest.approx(1.0, rel=0.01)

    def test_rel_mae_half_error(self):
        pred = torch.ones(1, 3, 8, 8) * 5.0
        target = torch.ones(1, 3, 8, 8) * 10.0
        mask = torch.ones(1, 1, 8, 8)
        m = compute_regression_metrics(pred, target, mask)
        # MAE = 5, sum(|target|) = 10 * 64 = 640
        # rel_mae = (5*64) / 640 = 0.5
        for ch in ("pressure", "u", "v"):
            assert m[f"{ch}_rel_mae"] == pytest.approx(0.5, rel=0.02)

    def test_rel_mae_not_negative(self):
        pred, target, mask = _make_tensors()
        m = compute_regression_metrics(pred, target, mask)
        for ch in ("pressure", "u", "v"):
            assert m[f"{ch}_rel_mae"] >= 0.0


# ======================================================================
# RMSE ≥ MAE (always true for RMS vs mean abs)
# ======================================================================

class TestRMSERelationship:
    def test_rmse_ge_mae(self):
        pred, target, mask = _make_tensors()
        m = compute_regression_metrics(pred, target, mask)
        for ch in ("pressure", "u", "v"):
            assert m[f"{ch}_rmse"] >= m[f"{ch}_mae"] - 1e-6, (
                f"RMSE should be >= MAE for {ch}"
            )
