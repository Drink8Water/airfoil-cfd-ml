"""Smoke tests for geometry encoding: coordinate grids, SDF, channel augmentation.

All tests use synthetic data — no real .npz files required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from airfoil_cfd_ml.data.geometry import (
    GEOMETRY_MODES,
    MODE_CHANNEL_MAP,
    augment_geometry_channels,
    build_coordinate_grid,
    get_input_channels,
)
from airfoil_cfd_ml.data.sdf import (
    compute_mask_statistics,
    mask_to_boundary_distance,
    mask_to_sdf,
)


# ======================================================================
# build_coordinate_grid
# ======================================================================

class TestCoordinateGrid:
    def test_shape(self):
        """Returns (2, H, W)."""
        g = build_coordinate_grid(64, 96)
        assert g.shape == (2, 64, 96)
        assert g.dtype == torch.float32

    def test_range(self):
        """Values are in [-1, 1]."""
        g = build_coordinate_grid(128, 128)
        assert g.min() >= -1.0
        assert g.max() <= 1.0

    def test_square_symmetry(self):
        """For square grid, centre pixel is approximately (0, 0)."""
        g = build_coordinate_grid(127, 127)  # odd → exact 0 at centre
        cx = g[0, 63, 63].item()
        cy = g[1, 63, 63].item()
        assert abs(cx) < 0.02
        assert abs(cy) < 0.02

    def test_corners_square(self):
        """Corners are at (±1, ±1)."""
        g = build_coordinate_grid(64, 64)
        # top-left
        assert g[0, 0, 0] == pytest.approx(-1.0, abs=0.1)
        assert g[1, 0, 0] == pytest.approx(-1.0, abs=0.1)
        # bottom-right
        assert g[0, 63, 63] == pytest.approx(1.0, abs=0.1)
        assert g[1, 63, 63] == pytest.approx(1.0, abs=0.1)

    def test_nonsquare(self):
        """Works with non-square grids."""
        g = build_coordinate_grid(32, 64)
        assert g.shape == (2, 32, 64)
        # x should span [-1,1] across width
        assert g[0, 0, 0] == pytest.approx(-1.0, abs=0.1)
        assert g[0, 0, 63] == pytest.approx(1.0, abs=0.1)


# ======================================================================
# mask_to_sdf
# ======================================================================

class TestMaskToSDF:
    @staticmethod
    def _disc_mask(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
        """Binary fluid mask with a circular solid disc."""
        yy, xx = np.ogrid[:h, :w]
        solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        return (~solid).astype(np.uint8)  # 1=fluid, 0=solid

    def test_sdf_fluid_positive(self):
        """SDF is positive in fluid region."""
        mask = self._disc_mask(64, 64, 31, 31, 10)
        sdf = mask_to_sdf(mask)
        fluid = mask.astype(bool)
        assert (sdf[fluid] > 0).all(), "SDF should be > 0 in fluid"
        assert (sdf[~fluid] < 0).all(), "SDF should be < 0 in solid"

    def test_sdf_zero_at_boundary(self):
        """SDF is approximately 0 at the fluid/solid boundary."""
        mask = self._disc_mask(64, 64, 31, 31, 10)
        sdf = mask_to_sdf(mask)
        # The boundary is where mask changes — SDF should be near zero there.
        # Check that min |sdf| among boundary pixels is small.
        from scipy.ndimage import binary_dilation, binary_erosion

        boundary = binary_dilation(mask) ^ binary_erosion(mask)
        if boundary.any():
            boundary_sdf_abs = np.abs(sdf[boundary])
            assert boundary_sdf_abs.min() < 1.5, (
                f"Boundary SDF should be near zero, got min={boundary_sdf_abs.min():.2f}"
            )

    def test_truncate(self):
        """Truncation clamps SDF to [-trunc, +trunc]."""
        mask = self._disc_mask(64, 64, 31, 31, 10)
        sdf = mask_to_sdf(mask, truncate=8.0)
        assert sdf.min() >= -8.0
        assert sdf.max() <= 8.0

    def test_no_truncate(self):
        """truncate=None gives full range."""
        mask = self._disc_mask(64, 64, 31, 31, 10)
        sdf = mask_to_sdf(mask, truncate=None)
        # Some distances will exceed typical truncation
        assert sdf.max() > 8.0 or sdf.min() < -8.0

    def test_dtype(self):
        """Output is float32."""
        mask = self._disc_mask(32, 32, 15, 15, 6)
        sdf = mask_to_sdf(mask)
        assert sdf.dtype == np.float32

    def test_raises_on_all_fluid(self):
        """All-fluid mask raises ValueError."""
        mask = np.ones((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError, match="no solid"):
            mask_to_sdf(mask)

    def test_raises_on_all_solid(self):
        """All-solid mask raises ValueError."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError, match="no fluid"):
            mask_to_sdf(mask)

    def test_raises_on_3d(self):
        """3D input raises ValueError."""
        with pytest.raises(ValueError, match="2D"):
            mask_to_sdf(np.zeros((3, 32, 32), dtype=np.uint8))


# ======================================================================
# mask_to_boundary_distance
# ======================================================================

class TestBoundaryDistance:
    @staticmethod
    def _disc_mask(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
        yy, xx = np.ogrid[:h, :w]
        solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        return (~solid).astype(np.uint8)

    def test_fluid_positive(self):
        """Distance is > 0 in fluid, 0 in solid."""
        mask = self._disc_mask(64, 64, 31, 31, 10)
        bd = mask_to_boundary_distance(mask)
        fluid = mask.astype(bool)
        assert (bd[fluid] >= 0).all()
        assert (bd[~fluid] == 0).all()

    def test_truncate(self):
        """Truncation works."""
        mask = self._disc_mask(64, 64, 31, 31, 10)
        bd = mask_to_boundary_distance(mask, truncate=8.0)
        assert bd.max() <= 8.0

    def test_dtype(self):
        """Output is float32."""
        mask = self._disc_mask(32, 32, 15, 15, 6)
        bd = mask_to_boundary_distance(mask)
        assert bd.dtype == np.float32


# ======================================================================
# compute_mask_statistics
# ======================================================================

class TestMaskStatistics:
    def test_fractions(self):
        """Reports correct fluid fraction."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:16, :] = 1  # half fluid
        stats = compute_mask_statistics(mask)
        assert stats["fluid_fraction"] == pytest.approx(0.5, abs=0.01)
        assert stats["total_pixels"] == 1024

    def test_isolated_flag(self):
        """Detects two islands of fluid."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        mask[20:24, 20:24] = 1
        stats = compute_mask_statistics(mask)
        assert stats["has_isolated_fluid"] is True

    def test_connected_ok(self):
        """Single connected fluid region is not flagged."""
        mask = np.ones((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 0
        stats = compute_mask_statistics(mask)
        assert stats["has_isolated_fluid"] is False


# ======================================================================
# augment_geometry_channels
# ======================================================================

class TestAugmentGeometryChannels:
    @staticmethod
    def _make_x(h: int = 64, w: int = 64) -> torch.Tensor:
        """Create a synthetic 3-channel input with a disc obstacle."""
        u_inf_x = torch.full((h, w), 0.8)
        u_inf_y = torch.full((h, w), 0.0)
        mask = torch.zeros((h, w))
        cy, cx = h // 2, w // 2
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        r = 10.0
        solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        mask[solid] = 1.0
        return torch.stack([u_inf_x, u_inf_y, mask], dim=0)

    def test_mask_only_identity(self):
        """mask_only returns input unchanged."""
        x = self._make_x()
        out = augment_geometry_channels(x, mode="mask_only")
        assert out.shape == x.shape
        assert torch.equal(out, x)

    def test_mask_xy_shape(self):
        """mask_xy adds 2 channels."""
        x = self._make_x()
        out = augment_geometry_channels(x, mode="mask_xy")
        assert out.shape == (5, 64, 64)
        # XY coords should be in [-1, 1]
        assert out[3].min() >= -1.0 and out[3].max() <= 1.0
        assert out[4].min() >= -1.0 and out[4].max() <= 1.0

    def test_mask_xy_sdf_shape(self):
        """mask_xy_sdf adds 3 channels (xy + sdf)."""
        x = self._make_x()
        out = augment_geometry_channels(x, mode="mask_xy_sdf")
        assert out.shape == (6, 64, 64)
        # SDF channel should be in [-1, 1] (normalised by truncate)
        assert out[5].min() >= -1.0 and out[5].max() <= 1.0

    def test_mask_xy_sdf_boundary_shape(self):
        """mask_xy_sdf_boundary adds 4 channels."""
        x = self._make_x()
        out = augment_geometry_channels(x, mode="mask_xy_sdf_boundary")
        assert out.shape == (7, 64, 64)
        assert out[6].min() >= 0.0 and out[6].max() <= 1.0

    def test_all_modes_produce_expected_channels(self):
        """Each mode produces the documented channel count."""
        x = self._make_x()
        for mode, expected_c in MODE_CHANNEL_MAP.items():
            out = augment_geometry_channels(x, mode=mode)
            assert out.shape[0] == expected_c, (
                f"{mode}: expected {expected_c} channels, got {out.shape[0]}"
            )

    def test_unknown_mode_raises(self):
        """Invalid mode raises ValueError."""
        x = self._make_x()
        with pytest.raises(ValueError, match="Unknown geometry_mode"):
            augment_geometry_channels(x, mode="invalid_mode")

    def test_wrong_input_channels_raises(self):
        """2-channel input raises ValueError."""
        x = torch.randn(2, 64, 64)
        with pytest.raises(ValueError, match="Expected 3 input channels"):
            augment_geometry_channels(x, mode="mask_xy")

    def test_2d_input_raises(self):
        """2D input raises ValueError."""
        with pytest.raises(ValueError, match="3D"):
            augment_geometry_channels(torch.randn(64, 64), mode="mask_only")

    def test_sdf_symmetric(self):
        """SDF is antisymmetric: fluid SDF ≈ 0 at boundary, negative in solid."""
        x = self._make_x()
        out = augment_geometry_channels(x, mode="mask_xy_sdf")
        sdf_ch = out[5]  # normalised
        # In fluid: SDF > 0  (positive after normalisation)
        mask_fluid = x[2] < 0.5
        # Check fluid SDF > 0, solid SDF < 0 (or zero at boundary)
        fluid_sdf = sdf_ch[mask_fluid]
        solid_sdf = sdf_ch[~mask_fluid]
        if fluid_sdf.numel() > 0:
            assert fluid_sdf.min() >= -1e-5, "Fluid SDF should be >= 0"
        if solid_sdf.numel() > 0:
            assert solid_sdf.max() <= 1e-5, "Solid SDF should be <= 0"

    def test_boundary_distance_range(self):
        """Boundary distance is in [0, 1] after normalisation."""
        x = self._make_x()
        out = augment_geometry_channels(x, mode="mask_xy_sdf_boundary")
        bdist_ch = out[6]
        assert bdist_ch.min() >= 0.0
        assert bdist_ch.max() <= 1.0


# ======================================================================
# get_input_channels
# ======================================================================

class TestGetInputChannels:
    def test_known_modes(self):
        """Returns correct counts for all modes."""
        assert get_input_channels("mask_only") == 3
        assert get_input_channels("mask_xy") == 5
        assert get_input_channels("mask_xy_sdf") == 6
        assert get_input_channels("mask_xy_sdf_boundary") == 7

    def test_unknown_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown geometry_mode"):
            get_input_channels("nonsense")
