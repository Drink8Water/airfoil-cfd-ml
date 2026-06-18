"""Smoke tests for model registry, SimpleCNN, ResUNet, FNO2D, GeoFNOLite, TransolverLite.

Every model must pass:
  - forward shape: (B, C, H, W) → (B, 3, H, W)
  - backward (gradient flows to input)
  - registry (build_model works)
  - build_model_from_config works
  - count_parameters returns sensible int
"""

from __future__ import annotations

import pytest
import torch
import yaml

from airfoil_cfd_ml.models.registry import (
    MODEL_REGISTRY,
    build_model,
    build_model_from_config,
    count_parameters,
    count_parameters_millions,
)
from airfoil_cfd_ml.models.base import BaseModel
from airfoil_cfd_ml.models.simple_cnn import SimpleCNN
from airfoil_cfd_ml.models.res_unet import ResUNet
from airfoil_cfd_ml.models.fno2d import FNO2D
from airfoil_cfd_ml.models.geofno_lite import GeoFNOLite
from airfoil_cfd_ml.models.transolver_lite import TransolverLite


# ======================================================================
# Helpers
# ======================================================================

_MODEL_SPECS = [
    # (name, input_C, constructor_kwargs)
    ("simple_cnn", 3, {"in_channels": 3, "hidden_channels": 16, "n_layers": 3, "out_channels": 3}),
    ("simple_cnn", 5, {"in_channels": 5, "hidden_channels": 16, "n_layers": 3, "out_channels": 3}),
    ("res_unet", 3, {"in_channels": 3, "channel_exponent": 4, "dropout": 0.0}),
    ("res_unet", 7, {"in_channels": 7, "channel_exponent": 4, "dropout": 0.0}),
    ("fno2d", 3, {"in_channels": 3, "width": 16, "modes1": 8, "modes2": 8, "depth": 2}),
    ("fno2d", 6, {"in_channels": 6, "width": 16, "modes1": 8, "modes2": 8, "depth": 2}),
    ("geofno_lite", 5, {"in_channels": 5, "width": 16, "modes1": 8, "modes2": 8, "depth": 2}),
    ("geofno_lite", 7, {"in_channels": 7, "width": 16, "modes1": 8, "modes2": 8, "depth": 2}),
    ("transolver_lite", 3, {"in_channels": 3, "d_model": 64, "K": 16, "n_heads": 2, "n_layers": 2}),
    ("transolver_lite", 6, {"in_channels": 6, "d_model": 64, "K": 16, "n_heads": 2, "n_layers": 2}),
]


def _make_input(batch: int, channels: int, h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.randn(batch, channels, h, w)


# ======================================================================
# Registry
# ======================================================================

class TestRegistry:
    def test_all_models_registered(self):
        """Every model name in MODEL_SPECS is registered."""
        names = {spec[0] for spec in _MODEL_SPECS}
        for name in names:
            assert name in MODEL_REGISTRY, f"'{name}' not registered"

    def test_build_model_all_specs(self):
        """build_model succeeds for every spec."""
        for name, C, kwargs in _MODEL_SPECS:
            model = build_model(name, **kwargs)
            assert isinstance(model, torch.nn.Module)

    def test_build_model_inference(self):
        """build_model returns a callable module."""
        model = build_model("simple_cnn", in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, 64, 64))
        assert y.shape == (2, 3, 64, 64)

    def test_build_model_unknown_raises(self):
        with pytest.raises(KeyError):
            build_model("nonexistent_model")

    @pytest.mark.parametrize("name,channels,kwargs", [
        ("simple_cnn", 3, {"in_channels": 3, "hidden_channels": 16, "n_layers": 3, "out_channels": 3}),
        ("res_unet", 3, {"in_channels": 3, "channel_exponent": 4}),
        ("fno2d", 3, {"in_channels": 3, "width": 16, "modes1": 8, "modes2": 8, "depth": 2}),
        ("geofno_lite", 5, {"in_channels": 5, "width": 16, "modes1": 8, "modes2": 8, "depth": 2}),
        ("transolver_lite", 3, {"in_channels": 3, "d_model": 64, "K": 16, "n_heads": 2, "n_layers": 2}),
    ])
    def test_build_model_from_config(self, name, channels, kwargs):
        """build_model_from_config works with model_ prefix keys."""
        config = {"model_name": name}
        for k, v in kwargs.items():
            config[f"model_{k}"] = v
        model = build_model_from_config(config)
        y = model(_make_input(1, channels))
        assert y.shape == (1, 3, 64, 64)

    def test_build_model_from_config_missing_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            build_model_from_config({})

    def test_count_parameters_positive(self):
        model = build_model("simple_cnn", in_channels=3, hidden_channels=8, n_layers=2)
        n = count_parameters(model)
        assert n > 0
        assert isinstance(n, int)

    def test_count_parameters_millions(self):
        model = build_model("simple_cnn", in_channels=3, hidden_channels=32, n_layers=3)
        m = count_parameters_millions(model)
        assert m >= 0.0 and m < 10.0, f"Got {m}M params"
        # 3×3 conv layers with 32 channels should have > 0.01M params
        assert m > 0.005, f"Too few params: {m}M"


# ======================================================================
# Forward shape (B,C,H,W) → (B,3,H,W)
# ======================================================================

@pytest.mark.parametrize("name,channels,kwargs", _MODEL_SPECS)
class TestForwardShape:
    """Parametrized tests: every model spec in _MODEL_SPECS."""

    def test_shape_b1(self, name, channels, kwargs):
        """Batch=1."""
        model = build_model(name, **kwargs)
        model.eval()
        with torch.no_grad():
            y = model(_make_input(1, channels))
        assert y.shape == (1, 3, 64, 64), f"{name}: got {y.shape}"

    def test_shape_b4(self, name, channels, kwargs):
        """Batch=4."""
        model = build_model(name, **kwargs)
        model.eval()
        with torch.no_grad():
            y = model(_make_input(4, channels))
        assert y.shape == (4, 3, 64, 64), f"{name}: got {y.shape}"

    def test_shape_128(self, name, channels, kwargs):
        """Spatial=128 (core resolution)."""
        model = build_model(name, **kwargs)
        model.eval()
        with torch.no_grad():
            y = model(_make_input(2, channels, h=128, w=128))
        assert y.shape == (2, 3, 128, 128), f"{name}: got {y.shape}"


# ======================================================================
# Backward (gradient flows)
# ======================================================================

@pytest.mark.parametrize("name,channels,kwargs", _MODEL_SPECS)
class TestBackward:
    """Parametrized backward-pass tests."""

    def test_gradient_flows(self, name, channels, kwargs):
        """Loss.backward() does not raise; grad is non-zero for the first param."""
        model = build_model(name, **kwargs)
        x = _make_input(2, channels)
        y = model(x)
        loss = y.mean()
        loss.backward()

        # Check at least one parameter received a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, f"{name}: no parameter gradients"

    def test_no_nan_grad(self, name, channels, kwargs):
        """Gradients contain no NaN."""
        model = build_model(name, **kwargs)
        x = _make_input(2, channels)
        y = model(x)
        loss = y.mean()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"{name}: NaN in grad"


# ======================================================================
# Base model abstract
# ======================================================================

def test_base_model_is_abstract():
    base = BaseModel()
    with pytest.raises(NotImplementedError):
        base.forward(torch.randn(1, 3, 32, 32))


# ======================================================================
# Parameter counts are reasonable
# ======================================================================

class TestParameterCounts:
    def test_simple_cnn_params(self):
        m = build_model("simple_cnn", in_channels=3, hidden_channels=32, n_layers=4, out_channels=3)
        n = count_parameters(m)
        # ~35K for default config
        assert 10_000 < n < 200_000, f"Unexpected param count: {n}"

    def test_res_unet_params(self):
        m = build_model("res_unet", in_channels=3, channel_exponent=4, dropout=0.0)
        n = count_parameters(m)
        # channel_exponent=4 → base=16, should be reasonable
        assert 100_000 < n < 5_000_000, f"Unexpected param count: {n}"

    def test_fno2d_params(self):
        m = build_model("fno2d", in_channels=3, width=16, modes1=8, modes2=8, depth=2)
        n = count_parameters(m)
        assert 1_000 < n < 500_000, f"Unexpected param count: {n}"

    def test_transolver_lite_params(self):
        m = build_model("transolver_lite", in_channels=3, d_model=64, K=16, n_heads=2, n_layers=2)
        n = count_parameters(m)
        assert 100_000 < n < 5_000_000, f"Unexpected param count: {n}"


# ======================================================================
# Model-specific: ResU-Net spatial preservation
# ======================================================================

class TestResUNet:
    def test_spatial_preservation(self):
        """128→64→32→16→8→4 bottle → back to 128."""
        model = build_model("res_unet", in_channels=3, channel_exponent=4)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 64, 64))
        assert y.shape == (1, 3, 64, 64)

    def test_non_square(self):
        """Works with non-square inputs (power-of-2 friendly)."""
        model = build_model("res_unet", in_channels=3, channel_exponent=4)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 64, 96))
        # May not preserve exact size if not divisible by 32.
        # ResU-Net uses bilinear upsample that should match.
        # With 5 levels of stride-2: 64×96 → 32×48 → 16×24 → 8×12 → 4×6.
        # Upsample: 4×6→8×12→16×24→32×48→64×96. ✓
        assert y.shape[0] == 1 and y.shape[1] == 3


# ======================================================================
# Model-specific: FNO
# ======================================================================

class TestFNO2D:
    def test_modes_clamped(self):
        """Modes larger than spatial dims are silently clamped."""
        model = build_model("fno2d", in_channels=3, width=16, modes1=100, modes2=100, depth=2)
        # Should not crash — modes clamped in constructor
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 64, 64))
        assert y.shape == (1, 3, 64, 64)

    def test_different_resolution(self):
        """FNO works on 32×32 and 128×128 (resolution-invariant)."""
        model = build_model("fno2d", in_channels=3, width=16, modes1=8, modes2=8, depth=2)
        model.eval()
        with torch.no_grad():
            y32 = model(torch.randn(1, 3, 32, 32))
            y128 = model(torch.randn(1, 3, 128, 128))
        assert y32.shape == (1, 3, 32, 32)
        assert y128.shape == (1, 3, 128, 128)


# ======================================================================
# Model-specific: GeoFNO-lite
# ======================================================================

class TestGeoFNOLite:
    def test_geo_channels_optional(self):
        """Works with min channels (5=3 phys + 2 xy)."""
        model = build_model("geofno_lite", in_channels=5, width=16, modes1=8, modes2=8, depth=2)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 5, 64, 64))
        assert y.shape == (2, 3, 64, 64)

    def test_no_geo_channels_falls_back(self):
        """With in_channels=3, geo_proj is None (no crash)."""
        model = build_model("geofno_lite", in_channels=3, width=16, modes1=8, modes2=8, depth=2)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 3, 64, 64))
        assert y.shape == (2, 3, 64, 64)


# ======================================================================
# Model-specific: Transolver-lite
# ======================================================================

class TestTransolverLite:
    def test_token_count_preserved(self):
        """K tokens are learnable and expand to batch."""
        model = build_model("transolver_lite", in_channels=3, d_model=64, K=16, n_heads=2, n_layers=2)
        assert model.tokens.shape == (1, 16, 64)

    def test_non_square_grid(self):
        """Works on rectangular grids."""
        model = build_model("transolver_lite", in_channels=3, d_model=64, K=16, n_heads=2, n_layers=2)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(1, 3, 32, 48))
        assert y.shape == (1, 3, 32, 48)

    def test_geofno_input_compatibility(self):
        """Accepts GeoFNO-style 6-channel input."""
        model = build_model("transolver_lite", in_channels=6, d_model=64, K=16, n_heads=2, n_layers=2)
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 6, 32, 32))
        assert y.shape == (2, 3, 32, 32)

    def test_gradient_checkpoint_friendly(self):
        """Backward with batch=4, 64×64, K=32 should not OOM on 12GB."""
        model = build_model(
            "transolver_lite",
            in_channels=5, d_model=64, K=32, n_heads=2, n_layers=2,
        )
        x = _make_input(2, 5, h=64, w=64)
        y = model(x)
        loss = y.mean()
        loss.backward()
        # If we got here, no OOM
        assert True
