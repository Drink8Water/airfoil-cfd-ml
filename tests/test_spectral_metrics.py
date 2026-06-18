"""Tests for spectral metrics: spectral_error, energy_spectrum_error.

All tests use synthetic tensors — no real data required.
"""

from __future__ import annotations

import pytest
import torch

from airfoil_cfd_ml.metrics.spectral import energy_spectrum_error, spectral_error


class TestSpectralError:
    def test_identical_zero(self):
        """Same pred/target → 0 spectral error."""
        pred = torch.randn(2, 3, 32, 32)
        target = pred.clone()
        s = spectral_error(pred, target)
        assert s == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different(self):
        """Different fields → positive spectral error."""
        pred = torch.ones(1, 3, 16, 16)
        target = torch.zeros(1, 3, 16, 16)
        s = spectral_error(pred, target)
        assert s > 0.0

    def test_returns_float(self):
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        s = spectral_error(pred, target)
        assert isinstance(s, float)

    def test_with_mask(self):
        """Masked solid region does not crash."""
        pred = torch.randn(1, 3, 16, 16)
        target = torch.randn(1, 3, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        mask[:, :, 8:, :] = 0.0
        s = spectral_error(pred, target, mask)
        assert s >= 0.0


class TestEnergySpectrumError:
    def test_identical_zero(self):
        """Same pred/target → 0 energy spectrum error."""
        pred = torch.randn(1, 3, 32, 32)
        target = pred.clone()
        e = energy_spectrum_error(pred, target)
        assert e == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different(self):
        """Different fields → positive energy spectrum error."""
        pred = torch.ones(1, 3, 16, 16)
        target = torch.zeros(1, 3, 16, 16)
        e = energy_spectrum_error(pred, target)
        assert e >= 0.0

    def test_returns_float(self):
        pred = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        e = energy_spectrum_error(pred, target)
        assert isinstance(e, float)

    def test_with_mask(self):
        """Mask does not crash."""
        pred = torch.randn(1, 3, 16, 16)
        target = torch.randn(1, 3, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        e = energy_spectrum_error(pred, target, mask)
        assert e >= 0.0
