"""Device resolution utility."""

from __future__ import annotations

import torch


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return torch.device('cuda') if available and requested, else 'cpu'."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
