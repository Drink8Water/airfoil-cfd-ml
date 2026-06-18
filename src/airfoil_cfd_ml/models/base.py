"""Base model abstract class.

All models MUST satisfy:
  forward(x) with x: (B, C, H, W) → y: (B, 3, H, W)
"""

from __future__ import annotations

import torch.nn as nn


class BaseModel(nn.Module):
    """Abstract base for all surrogate models.

    Subclasses must implement forward(x) → (B, 3, H, W).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward().")
