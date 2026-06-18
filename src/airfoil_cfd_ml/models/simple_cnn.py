"""SimpleCNN: a minimal conv-net baseline for smoke testing the framework.

Architecture:
  N conv layers (3x3, same padding) with BatchNorm + ReLU,
  final 1x1 conv to 3 output channels.

Configurable: in_channels, hidden_channels, n_layers, out_channels.
"""

from __future__ import annotations

import torch.nn as nn

from .base import BaseModel
from .registry import register_model


@register_model("simple_cnn")
class SimpleCNN(BaseModel):
    """Stacked conv layers with BatchNorm and ReLU, preserving spatial size.

    Args:
        in_channels: input channel count (default 3).
        hidden_channels: channel width for hidden layers (default 32).
        n_layers: total conv layers including final projection (default 4).
        out_channels: output channel count (default 3).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        n_layers: int = 4,
        out_channels: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for i in range(n_layers - 1):
            layers.extend(
                [
                    nn.Conv2d(
                        prev,
                        hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            prev = hidden_channels
        # Final projection (no BatchNorm, no activation)
        layers.append(nn.Conv2d(prev, out_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            (B, 3, H, W) output tensor.
        """
        return self.net(x)
