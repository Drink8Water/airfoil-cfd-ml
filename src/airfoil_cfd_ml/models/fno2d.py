"""FNO2D: Fourier Neural Operator for 2D spatial fields.

Implements SpectralConv2d using torch.fft.rfft2 / irfft2 with learnable
complex weights on truncated Fourier modes.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations", ICLR 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class SpectralConv2d(nn.Module):
    """2D spectral convolution: learnable weights on truncated Fourier modes.

    Args:
        in_channels: input channels.
        out_channels: output channels.
        modes1: number of Fourier modes along dim=-2 (height).
        modes2: number of Fourier modes along dim=-1 (width).
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Complex weights: (in_ch, out_ch, modes1, modes2), dtype=complex64
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, input_complex: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Batched complex multiplication (element-wise per mode).

        Args:
            input_complex:  (B, in_ch, modes1, modes2)  complex
            weights:        (in_ch, out_ch, modes1, modes2) complex

        Returns:
            (B, out_ch, modes1, modes2) complex
        """
        # input_complex: (B, in, modes1, modes2)
        # weights:        (in, out, modes1, modes2)
        # → (B, in, out, modes1, modes2) via einsum
        return torch.einsum("bixy,ioxy->boxy", input_complex, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) real input.

        Returns:
            (B, C_out, H, W) real output (spectral part only).
        """
        B, C, H, W = x.shape

        # Effective modes: can't exceed the frequency content of the input
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        # rfft2 along last two dims → (B, C, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Truncate to effective modes
        x_ft_trunc = x_ft[:, :, :m1, :m2]  # (B, C, m1, m2)

        # Slice weights to match
        w = self.weights[:, :, :m1, :m2]  # (in, out, m1, m2)

        # Complex linear transform
        out_ft = self._compl_mul2d(x_ft_trunc, w)  # (B, C_out, m1, m2)

        # Zero-pad back to full spectral shape
        out_ft_full = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft_full[:, :, :m1, :m2] = out_ft

        # irfft2
        out = torch.fft.irfft2(out_ft_full, s=(H, W), norm="ortho")
        return out  # (B, C_out, H, W)


class FNOBlock(nn.Module):
    """One FNO layer: SpectralConv2d + linear skip + activation."""

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.spec_conv = SpectralConv2d(width, width, modes1, modes2)
        self.conv = nn.Conv2d(width, width, 1)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spec_conv(x)
        x2 = self.conv(x)
        return self.act(x1 + x2)


@register_model("fno2d")
class FNO2D(nn.Module):
    """Fourier Neural Operator for 2D airfoil flow prediction.

    Args:
        in_channels: input channel count (depends on geometry_mode).
        width: hidden channel width (default 32).
        modes1: Fourier modes along height (default 12 for 128×128).
        modes2: Fourier modes along width (default 12 for 128×128).
        depth: number of FNO blocks (default 4).
        activation: "gelu" or "relu".
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.modes1 = min(modes1, 128)
        self.modes2 = min(modes2, 65)  # W//2+1 for W=128

        # Lifting
        self.lift = nn.Conv2d(in_channels, width, 1)

        # FNO layers
        self.layers = nn.ModuleList([
            FNOBlock(width, self.modes1, self.modes2, activation)
            for _ in range(depth)
        ])

        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W).

        Returns:
            (B, 3, H, W).
        """
        x = self.lift(x)

        for layer in self.layers:
            x = layer(x)

        return self.project(x)
