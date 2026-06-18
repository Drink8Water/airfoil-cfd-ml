"""GeoFNO-lite: FNO backbone with explicit geometry channels.

Inspired by Li et al., "Geometry-Informed Neural Operator for Large-Scale
3D PDEs", NeurIPS 2023.  **This is NOT an official reproduction.**  It is
a lightweight variant that augments FNO2D input with pre-computed geometry
channels (xy coordinates, SDF, boundary distance) rather than learning a
coordinate transform inside the operator.

Recommended input: 5–7 channels via AirfoilNPZDataset geometry_mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .fno2d import FNO2D
from .registry import register_model


@register_model("geofno_lite")
class GeoFNOLite(nn.Module):
    """Geometry-aware FNO-lite for airfoil flow prediction.

    Wraps FNO2D with a geometry projection MLP.  Geometry channels (beyond
    the 3 physical channels) are processed by a small MLP and added to the
    lifted features before entering the FNO backbone.

    Args:
        in_channels: total input channels (e.g. 5 for mask_xy, 6 for
                     mask_xy_sdf, 7 for mask_xy_sdf_boundary).
        width: FNO hidden width.
        modes1: Fourier modes along height.
        modes2: Fourier modes along width.
        depth: number of FNO blocks.
        geo_mlp_hidden: hidden dim for the geometry-projection MLP.
        activation: "gelu" or "relu".
    """

    def __init__(
        self,
        in_channels: int = 6,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 4,
        geo_mlp_hidden: int = 32,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width

        # Number of geometry channels (everything beyond the 3 physical)
        n_geo = max(0, in_channels - 3)

        # Geometry MLP: 1×1 conv that maps geo channels → width,
        # so we can add geometry features to lifted physical features.
        self.geo_proj = (
            nn.Sequential(
                nn.Conv2d(n_geo, geo_mlp_hidden, 1),
                nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True),
                nn.Conv2d(geo_mlp_hidden, width, 1),
            )
            if n_geo > 0
            else None
        )

        # Physical lifting
        self.phys_lift = nn.Conv2d(3, width, 1)

        # FNO backbone
        self.fno = FNO2D(
            in_channels=width,
            width=width,
            modes1=modes1,
            modes2=modes2,
            depth=depth,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) where C = in_channels.
               Channels 0-2: physical [u_inf_x, u_inf_y, mask].
               Channels 3+: geometry (xy, SDF, boundary distance).

        Returns:
            (B, 3, H, W).
        """
        phys = x[:, :3, :, :]   # (B, 3, H, W)
        geo = x[:, 3:, :, :]    # (B, C-3, H, W)

        # Lift physical channels
        h = self.phys_lift(phys)  # (B, width, H, W)

        # Add geometry contribution (if any)
        if self.geo_proj is not None and geo.shape[1] > 0:
            h = h + self.geo_proj(geo)

        return self.fno(h)
