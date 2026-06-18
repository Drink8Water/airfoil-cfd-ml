"""Smoke tests for data schema, DataSample, collate_stack, and validate_npz_file."""

import os
import tempfile

import numpy as np
import pytest
import torch

from airfoil_cfd_ml.data.schema import (
    CANONICAL_KEY,
    DataSample,
    collate_stack,
    validate_npz_file,
)


def test_data_sample_creation():
    """DataSample stores x, y, mask, meta correctly."""
    s = DataSample(
        x=torch.randn(3, 128, 128),
        y=torch.randn(3, 128, 128),
        mask=torch.ones(1, 128, 128),
        meta={"file": "sample_001.npz", "index": 0},
    )
    assert s.x.shape == (3, 128, 128)
    assert s.y.shape == (3, 128, 128)
    assert s.mask.shape == (1, 128, 128)
    assert s.meta["file"] == "sample_001.npz"
    assert s.meta["index"] == 0


def test_data_sample_default_meta():
    """DataSample meta defaults to empty dict."""
    s = DataSample(
        x=torch.randn(3, 64, 64),
        y=torch.randn(3, 64, 64),
        mask=torch.zeros(1, 64, 64),
    )
    assert s.meta == {}


def test_collate_stack_shapes():
    """collate_stack batches N samples along dim=0."""
    samples = [
        DataSample(
            x=torch.randn(3, 128, 128),
            y=torch.randn(3, 128, 128),
            mask=torch.ones(1, 128, 128),
            meta={},
        ),
        DataSample(
            x=torch.randn(3, 128, 128),
            y=torch.randn(3, 128, 128),
            mask=torch.ones(1, 128, 128),
            meta={},
        ),
        DataSample(
            x=torch.randn(3, 128, 128),
            y=torch.randn(3, 128, 128),
            mask=torch.ones(1, 128, 128),
            meta={},
        ),
    ]
    batch = collate_stack(samples)
    assert batch.x.shape == (3, 3, 128, 128)
    assert batch.y.shape == (3, 3, 128, 128)
    assert batch.mask.shape == (3, 1, 128, 128)
    assert batch.meta["n_samples"] == 3


def test_collate_stack_preserves_values():
    """collate_stack preserves first sample's values."""
    s0 = DataSample(
        x=torch.ones(2, 4, 4),
        y=torch.zeros(3, 4, 4),
        mask=torch.full((1, 4, 4), 0.5),
        meta={},
    )
    s1 = DataSample(
        x=torch.ones(2, 4, 4) * 2,
        y=torch.zeros(3, 4, 4),
        mask=torch.full((1, 4, 4), 0.5),
        meta={},
    )
    batch = collate_stack([s0, s1])
    assert torch.allclose(batch.x[0], torch.ones(2, 4, 4))
    assert torch.allclose(batch.x[1], torch.ones(2, 4, 4) * 2)


# ======================================================================
# Helpers for validate_npz_file tests
# ======================================================================

def _make_valid_npz(path: str, h: int = 128, w: int = 128) -> None:
    """Write a canonical valid .npz file."""
    arr = np.zeros((6, h, w), dtype=np.float32)
    arr[0] = 1.0  # u_inf_x
    arr[1] = 0.0  # u_inf_y
    arr[2] = np.where(
        (np.arange(h)[:, None] - h // 2) ** 2
        + (np.arange(w)[None, :] - w // 2) ** 2
        <= 20**2,
        1.0,
        0.0,
    ).astype(np.float32)  # binary mask with disc obstacle
    arr[3] = np.random.randn(h, w).astype(np.float32) * 0.1  # pressure
    arr[4] = np.random.randn(h, w).astype(np.float32) * 10.0  # u_flow
    arr[5] = np.random.randn(h, w).astype(np.float32) * 5.0  # v_flow
    np.savez_compressed(path, **{CANONICAL_KEY: arr})


# ======================================================================
# validate_npz_file
# ======================================================================

class TestValidateNpzFile:
    def test_valid_file_passes(self):
        """A well-formed file returns valid=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample_000.npz")
            _make_valid_npz(path)
            result = validate_npz_file(path)
            assert result["valid"] is True
            assert result["errors"] == []
            assert result["shape"] == (6, 128, 128)
            assert result["dtype"] == "float32"

    def test_valid_file_with_nonstandard_spatial_warns(self):
        """Non-128x128 spatial produces a warning but remains valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample_000.npz")
            _make_valid_npz(path, h=64, w=64)
            result = validate_npz_file(path, expected_spatial=(128, 128))
            assert result["valid"] is True
            assert any("Spatial shape" in w for w in result["warnings"])

    def test_spatial_none_skips_check(self):
        """expected_spatial=None accepts any spatial size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample_000.npz")
            _make_valid_npz(path, h=64, w=64)
            result = validate_npz_file(path, expected_spatial=None)
            assert result["valid"] is True

    def test_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_npz_file("/nonexistent/path/file.npz")

    def test_missing_key(self):
        """A .npz without key 'a' is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            np.savez_compressed(path, b=np.zeros((6, 128, 128)))
            result = validate_npz_file(path)
            assert result["valid"] is False
            assert any("Missing key" in e for e in result["errors"])

    def test_wrong_channels(self):
        """4-channel array is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            np.savez_compressed(path, **{CANONICAL_KEY: np.zeros((4, 128, 128), dtype=np.float32)})
            result = validate_npz_file(path)
            assert result["valid"] is False
            assert any("Expected 6 channels" in e for e in result["errors"])

    def test_wrong_ndim(self):
        """2D array is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            np.savez_compressed(path, **{CANONICAL_KEY: np.zeros((128, 128), dtype=np.float32)})
            result = validate_npz_file(path)
            assert result["valid"] is False

    def test_nan_detected(self):
        """NaN in array makes it invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            arr = np.zeros((6, 128, 128), dtype=np.float32)
            arr[0, 10, 10] = np.nan
            np.savez_compressed(path, **{CANONICAL_KEY: arr})
            result = validate_npz_file(path)
            assert result["valid"] is False
            assert any("NaN" in e for e in result["errors"])

    def test_inf_detected(self):
        """Inf in array makes it invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            arr = np.zeros((6, 128, 128), dtype=np.float32)
            arr[1, 20, 20] = np.inf
            np.savez_compressed(path, **{CANONICAL_KEY: arr})
            result = validate_npz_file(path)
            assert result["valid"] is False
            assert any("Inf" in e for e in result["errors"])

    def test_non_float32_warns(self):
        """float64 dtype warns but is still valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            arr = np.zeros((6, 128, 128), dtype=np.float64)
            arr[2] = np.where(
                (np.arange(128)[:, None] - 64) ** 2
                + (np.arange(128)[None, :] - 64) ** 2
                <= 20**2,
                1.0,
                0.0,
            )
            np.savez_compressed(path, **{CANONICAL_KEY: arr})
            result = validate_npz_file(path)
            assert result["valid"] is True
            assert any("dtype" in w for w in result["warnings"])

    def test_non_binary_mask_warns(self):
        """Mask with non-binary values warns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            arr = np.zeros((6, 128, 128), dtype=np.float32)
            arr[2] = np.random.rand(128, 128)  # continuous, not binary
            np.savez_compressed(path, **{CANONICAL_KEY: arr})
            result = validate_npz_file(path)
            assert result["mask_stats"] is not None
            assert result["mask_stats"]["is_binary"] is False
            assert any("binary" in w for w in result["warnings"])

    def test_binary_mask_passes(self):
        """Proper binary mask has is_binary=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ok.npz")
            _make_valid_npz(path)
            result = validate_npz_file(path)
            assert result["mask_stats"] is not None
            assert result["mask_stats"]["is_binary"] is True

    def test_corrupt_npz_handled(self):
        """A corrupt file is caught gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrupt.npz")
            with open(path, "wb") as f:
                f.write(b"not a valid npz file")
            result = validate_npz_file(path)
            assert result["valid"] is False
            assert any("Cannot load" in e for e in result["errors"])
