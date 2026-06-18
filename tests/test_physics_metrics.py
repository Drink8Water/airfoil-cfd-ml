"""Tests for physics-aware metrics: divergence, vorticity, boundary, wake.

All tests use synthetic tensors — no real data required.
"""

from __future__ import annotations

import pytest
import torch

from airfoil_cfd_ml.metrics.physics import (
    boundary_rel_mae,
    divergence_error,
    finite_diff_x,
    finite_diff_y,
    vorticity_error,
    wake_rel_mae,
)


# ======================================================================
# Helpers
# ======================================================================

def _disc_mask(b: int, h: int, w: int) -> torch.Tensor:
    """(B, 1, H, W) fluid mask with a circular solid obstacle at centre."""
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy, r = w // 2, h // 2, 8
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    mask = (~solid).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return mask.expand(b, 1, h, w)


# ======================================================================
# finite_diff
# ======================================================================

class TestFiniteDiff:
    def test_diff_x_linear(self):
        """∂x of linear ramp = constant = 1."""
        x = torch.arange(64, dtype=torch.float32).view(1, 1, 1, 64).expand(2, 3, 32, 64)
        dx = finite_diff_x(x)
        # Interior should be ~1.0
        assert dx[0, 0, 0, 5:60].mean().item() == pytest.approx(1.0, abs=0.1)

    def test_diff_y_linear(self):
        """∂y of linear ramp = constant = 1."""
        y = torch.arange(32, dtype=torch.float32).view(1, 1, 32, 1).expand(2, 3, 32, 64)
        dy = finite_diff_y(y)
        assert dy[0, 0, 5:25, 5].mean().item() == pytest.approx(1.0, abs=0.1)

    def test_diff_x_constant_zero(self):
        """∂x of constant field = 0."""
        u = torch.ones(2, 3, 32, 32)
        dx = finite_diff_x(u)
        assert dx.abs().max().item() < 1e-4

    def test_diff_y_constant_zero(self):
        """∂y of constant field = 0."""
        u = torch.ones(2, 3, 32, 32)
        dy = finite_diff_y(u)
        assert dy.abs().max().item() < 1e-4


# ======================================================================
# divergence_error
# ======================================================================

class TestDivergenceError:
    def test_zero_divergence_uniform_flow(self):
        """Uniform u=1, v=0 → divergence=0."""
        pred = torch.zeros(2, 3, 32, 32)
        pred[:, 1] = 1.0  # u=1
        pred[:, 2] = 0.0  # v=0
        mask = torch.ones(2, 1, 32, 32)
        d = divergence_error(pred, mask)
        assert d == pytest.approx(0.0, abs=1e-3)

    def test_positive_for_divergent_field(self):
        """u(x)=x, v(y)=0 → ∂u/∂x=1, div=1."""
        pred = torch.zeros(1, 3, 32, 32)
        xx = torch.arange(32, dtype=torch.float32).view(1, 1, 32).expand(1, 1, 32, 32)
        pred[:, 1:2] = xx
        d = divergence_error(pred)
        assert d > 0.5  # approximately 1.0

    def test_with_mask(self):
        """Masked regions excluded."""
        pred = torch.zeros(2, 3, 32, 32)
        ramp = torch.arange(32, dtype=torch.float32).view(1, 1, 1, 32).expand(2, 1, 32, 32)
        pred[:, 1:2] = ramp
        mask = torch.ones(2, 1, 32, 32)
        mask[:, :, :, 16:] = 0.0
        d_full = divergence_error(pred, torch.ones_like(mask))
        d_partial = divergence_error(pred, mask)
        assert abs(d_full - d_partial) < 0.5

    def test_3d_mask_accepted(self):
        """(B, H, W) mask is auto-unsqueezed."""
        pred = torch.zeros(1, 3, 16, 16)
        pred[:, 1] = 1.0
        mask = torch.ones(1, 16, 16)
        d = divergence_error(pred, mask)
        assert d == pytest.approx(0.0, abs=1e-3)


# ======================================================================
# vorticity_error
# ======================================================================

class TestVorticityError:
    def test_irrotational_uniform_zero(self):
        """Uniform flow → ω=0."""
        pred = torch.zeros(2, 3, 32, 32)
        pred[:, 1] = 1.0
        pred[:, 2] = 0.0
        target = pred.clone()
        v = vorticity_error(pred, target)
        assert v == pytest.approx(0.0, abs=1e-3)

    def test_positive_for_rotational(self):
        """v(x)=x, u=0 → ω = ∂v/∂x - ∂u/∂y = 1."""
        pred = torch.zeros(1, 3, 32, 32)
        xx = torch.arange(32, dtype=torch.float32).view(1, 1, 32).expand(1, 1, 32, 32)
        pred[:, 2:3] = xx
        v = vorticity_error(pred, target=None)
        assert v > 0.5

    def test_mae_mode_with_target(self):
        """Same pred/target → zero MAE."""
        pred = torch.randn(2, 3, 32, 32)
        target = pred.clone()
        v = vorticity_error(pred, target)
        assert v == pytest.approx(0.0, abs=1e-3)


# ======================================================================
# boundary_rel_mae
# ======================================================================

class TestBoundaryRelMAE:
    def test_returns_float(self):
        pred = torch.randn(2, 3, 64, 64)
        target = pred.clone()
        mask = _disc_mask(2, 64, 64)
        b = boundary_rel_mae(pred, target, mask)
        assert isinstance(b, float)
        assert b >= 0.0

    def test_perfect_pred_zero(self):
        pred = torch.ones(1, 3, 32, 32)
        target = pred.clone()
        mask = _disc_mask(1, 32, 32)
        b = boundary_rel_mae(pred, target, mask)
        assert b == pytest.approx(0.0, abs=1e-3)


# ======================================================================
# wake_rel_mae
# ======================================================================

class TestWakeRelMAE:
    def test_returns_float(self):
        pred = torch.randn(2, 3, 64, 64)
        target = pred.clone()
        mask = _disc_mask(2, 64, 64)
        w = wake_rel_mae(pred, target, mask)
        assert isinstance(w, float)
        assert w >= 0.0

    def test_perfect_pred_zero(self):
        pred = torch.ones(1, 3, 64, 64)
        target = pred.clone()
        mask = _disc_mask(1, 64, 64)
        w = wake_rel_mae(pred, target, mask)
        assert w == pytest.approx(0.0, abs=1e-3)
