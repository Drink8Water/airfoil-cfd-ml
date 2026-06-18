"""Dedicated tests for SpectralConv2d and FNO2D internals."""

from __future__ import annotations

import pytest
import torch

from airfoil_cfd_ml.models.fno2d import FNO2D, FNOBlock, SpectralConv2d


# ======================================================================
# SpectralConv2d
# ======================================================================

class TestSpectralConv2d:
    def test_shape_preserved(self):
        """Output has expected shape."""
        conv = SpectralConv2d(in_channels=8, out_channels=16, modes1=6, modes2=6)
        x = torch.randn(2, 8, 64, 64)
        y = conv(x)
        assert y.shape == (2, 16, 64, 64)

    def test_different_spatial(self):
        """Works on 32×32 and 128×128."""
        conv = SpectralConv2d(in_channels=4, out_channels=8, modes1=4, modes2=4)
        for hw in [(32, 32), (64, 96), (128, 128)]:
            y = conv(torch.randn(1, 4, *hw))
            assert y.shape == (1, 8, *hw), f"Failed on {hw}"

    def test_weights_complex(self):
        """Weights are complex-valued."""
        conv = SpectralConv2d(4, 8, 6, 6)
        assert conv.weights.dtype == torch.cfloat
        assert conv.weights.shape == (4, 8, 6, 6)

    def test_real_input_real_output(self):
        """Real input → real output."""
        conv = SpectralConv2d(4, 4, 8, 8)
        x = torch.randn(2, 4, 64, 64)
        y = conv(x)
        assert not y.is_complex()
        assert y.dtype == torch.float32

    def test_gradient_flows(self):
        """Gradient flows through SpectralConv2d."""
        conv = SpectralConv2d(4, 4, 6, 6)
        x = torch.randn(2, 4, 32, 32, requires_grad=True)
        y = conv(x)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_modes_larger_than_spatial(self):
        """When modes > spatial, truncation is safe."""
        conv = SpectralConv2d(4, 4, 100, 100)  # modes > H, W
        x = torch.randn(1, 4, 16, 16)
        y = conv(x)
        assert y.shape == (1, 4, 16, 16)


# ======================================================================
# FNOBlock
# ======================================================================

class TestFNOBlock:
    def test_shape(self):
        block = FNOBlock(width=16, modes1=8, modes2=8, activation="gelu")
        x = torch.randn(2, 16, 64, 64)
        y = block(x)
        assert y.shape == x.shape

    def test_gradient(self):
        block = FNOBlock(width=16, modes1=8, modes2=8)
        x = torch.randn(2, 16, 32, 32, requires_grad=True)
        y = block(x)
        loss = y.mean()
        loss.backward()
        assert x.grad is not None


# ======================================================================
# FNO2D
# ======================================================================

class TestFNO2DEndToEnd:
    def test_lifting_projection(self):
        """Input C → width → depth layers → 3."""
        model = FNO2D(in_channels=3, width=32, modes1=12, modes2=12, depth=4)
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        assert y.shape == (1, 3, 64, 64)

    def test_resolution_invariance(self):
        """Same FNO works on 32, 64, 128 spatial dims."""
        model = FNO2D(in_channels=3, width=16, modes1=8, modes2=8, depth=2)
        model.eval()
        with torch.no_grad():
            for hw in [(32, 32), (48, 64), (64, 96), (128, 128)]:
                y = model(torch.randn(1, 3, *hw))
                assert y.shape == (1, 3, *hw), f"Failed on {hw}"

    def test_deeper_network(self):
        """Deeper FNO (depth=6) runs without error."""
        model = FNO2D(in_channels=3, width=16, modes1=8, modes2=8, depth=6)
        y = model(torch.randn(1, 3, 64, 64))
        assert y.shape == (1, 3, 64, 64)

    def test_narrow_network(self):
        """Very narrow (width=8) works."""
        model = FNO2D(in_channels=3, width=8, modes1=8, modes2=8, depth=2)
        y = model(torch.randn(2, 3, 64, 64))
        assert y.shape == (2, 3, 64, 64)

    def test_spectral_weights_update(self):
        """Spectral weights change after one optimizer step."""
        model = FNO2D(in_channels=3, width=16, modes1=8, modes2=8, depth=2)
        w_before = model.layers[0].spec_conv.weights.data.clone()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(2, 3, 64, 64)
        loss = model(x).mean()
        loss.backward()
        opt.step()
        w_after = model.layers[0].spec_conv.weights.data
        assert not torch.allclose(w_before, w_after), "Spectral weights did not update"

    def test_no_nan_output(self):
        """Output contains no NaN with random weights."""
        model = FNO2D(in_channels=3, width=32, modes1=12, modes2=12, depth=4)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(4, 3, 128, 128))
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
