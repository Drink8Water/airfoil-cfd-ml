"""Dedicated tests for FieldLoss: mae, mse, huber, no NaN, backward.

Uses synthetic tensors — no real data required.
"""

from __future__ import annotations

import pytest
import torch

from airfoil_cfd_ml.losses.field import FieldLoss
from airfoil_cfd_ml.losses.composite import CompositeLoss


# ======================================================================
# Helpers
# ======================================================================

def _make_tensors(b: int = 2, c: int = 3, h: int = 64, w: int = 64):
    pred = torch.randn(b, c, h, w)
    target = torch.randn(b, c, h, w)
    mask = torch.ones(b, 1, h, w)
    return pred, target, mask


# ======================================================================
# FieldLoss — reduction types
# ======================================================================

class TestFieldLossReduction:
    def test_mse_positive(self):
        loss_fn = FieldLoss(reduction="mse")
        pred, target, mask = _make_tensors()
        total, d = loss_fn(pred, target, mask)
        assert total.item() > 0.0

    def test_mae_positive(self):
        loss_fn = FieldLoss(reduction="mae")
        pred, target, mask = _make_tensors()
        total, d = loss_fn(pred, target, mask)
        assert total.item() > 0.0

    def test_mse_zero_for_identical(self):
        loss_fn = FieldLoss(reduction="mse")
        pred = torch.ones(2, 3, 16, 16)
        target = pred.clone()
        mask = torch.ones(2, 1, 16, 16)
        total, d = loss_fn(pred, target, mask)
        assert total.item() == pytest.approx(0.0, abs=1e-5)

    def test_mae_zero_for_identical(self):
        loss_fn = FieldLoss(reduction="mae")
        pred = torch.ones(2, 3, 16, 16)
        target = pred.clone()
        mask = torch.ones(2, 1, 16, 16)
        total, d = loss_fn(pred, target, mask)
        assert total.item() == pytest.approx(0.0, abs=1e-5)

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction"):
            FieldLoss(reduction="huber")


# ======================================================================
# FieldLoss — mask
# ======================================================================

class TestFieldLossMask:
    def test_all_solid_zero_loss(self):
        loss_fn = FieldLoss(reduction="mse")
        pred = torch.randn(1, 3, 8, 8)
        target = pred.clone() + 5.0  # large error
        mask = torch.zeros(1, 1, 8, 8)  # all solid
        total, d = loss_fn(pred, target, mask)
        assert total.item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_mask(self):
        loss_fn = FieldLoss(reduction="mse")
        pred = torch.zeros(1, 3, 4, 4)
        target = torch.ones(1, 3, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        mask[:, :, 2:, :] = 0.0  # half masked out
        total, d = loss_fn(pred, target, mask)
        # Loss normalises by unmasked pixel count, so MSE ≈ 1.0 regardless
        # of the fraction masked (only the unmasked pixels contribute).
        assert 0.8 < total.item() < 1.2


# ======================================================================
# FieldLoss — channel weights
# ======================================================================

class TestFieldLossChannelWeights:
    def test_pressure_weighted(self):
        """Pressure weight=5 affects total loss; loss_dict reports unweighted.

        The per-channel values in loss_dict are from unweighted per_element
        (for monitoring), while the actual total loss uses channel_weights.
        """
        loss_fn = FieldLoss(
            reduction="mse", channel_weights=(5.0, 1.0, 1.0)
        )
        pred = torch.zeros(1, 3, 8, 8)
        target = torch.ones(1, 3, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        total, d = loss_fn(pred, target, mask)
        # Total should reflect weighted average: (5+1+1)/3 ≈ 2.33× unweighted
        assert total.item() > 1.5, f"Expected weighted total > 1.5, got {total.item():.3f}"
        # Unweighted per-channel values should be ~1.0 each
        for key in ("loss_pressure", "loss_u", "loss_v"):
            assert 0.8 < d[key] < 1.2, f"{key} should be ~1.0, got {d[key]:.3f}"


# ======================================================================
# FieldLoss — backward
# ======================================================================

class TestFieldLossBackward:
    def test_mse_gradient_flows(self):
        loss_fn = FieldLoss(reduction="mse")
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randn(2, 3, 16, 16)
        mask = torch.ones(2, 1, 16, 16)
        total, _d = loss_fn(pred, target, mask)
        total.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_mae_gradient_flows(self):
        loss_fn = FieldLoss(reduction="mae")
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randn(2, 3, 16, 16)
        mask = torch.ones(2, 1, 16, 16)
        total, _d = loss_fn(pred, target, mask)
        total.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_no_nan_grad(self):
        for reduction in ("mse", "mae"):
            loss_fn = FieldLoss(reduction=reduction)
            pred = torch.randn(2, 3, 16, 16, requires_grad=True)
            target = torch.randn(2, 3, 16, 16)
            mask = torch.ones(2, 1, 16, 16)
            total, _d = loss_fn(pred, target, mask)
            total.backward()
            assert not torch.isnan(pred.grad).any(), f"NaN in grad for {reduction}"


# ======================================================================
# FieldLoss — returns loss_dict keys
# ======================================================================

class TestFieldLossDict:
    def test_required_keys(self):
        loss_fn = FieldLoss(reduction="mse")
        pred, target, mask = _make_tensors()
        _total, d = loss_fn(pred, target, mask)
        for key in ("total", "loss_pressure", "loss_u", "loss_v"):
            assert key in d, f"Missing key: {key}"
            assert isinstance(d[key], float)

    def test_mae_values_positive(self):
        loss_fn = FieldLoss(reduction="mae")
        pred, target, mask = _make_tensors()
        _total, d = loss_fn(pred, target, mask)
        for v in d.values():
            assert v >= 0.0


# ======================================================================
# CompositeLoss
# ======================================================================

class TestCompositeLoss:
    def test_single_sub_loss(self):
        l1 = FieldLoss(reduction="mse")
        comp = CompositeLoss([(l1, 1.0)])
        pred, target, mask = _make_tensors()
        total, d = comp(pred, target, mask)
        assert "total" in d
        assert total.item() > 0.0

    def test_weighted_sum(self):
        l1 = FieldLoss(reduction="mse", channel_weights=(1.0, 1.0, 1.0))
        l2 = FieldLoss(reduction="mae", channel_weights=(1.0, 1.0, 1.0))
        # MSE only
        pred, target, mask = _make_tensors()
        total1, _ = l1(pred, target, mask)
        # MSE + 0*MAE
        comp = CompositeLoss([(l1, 1.0), (l2, 0.0)])
        total2, _ = comp(pred, target, mask)
        # Should be approximately equal
        assert total1.item() == pytest.approx(total2.item(), rel=0.01)

    def test_backward(self):
        l1 = FieldLoss(reduction="mse")
        comp = CompositeLoss([(l1, 1.0)])
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randn(2, 3, 16, 16)
        mask = torch.ones(2, 1, 16, 16)
        total, _ = comp(pred, target, mask)
        total.backward()
        assert pred.grad is not None
