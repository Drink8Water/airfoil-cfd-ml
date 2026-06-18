"""Deterministic seed utilities for reproducible training."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    """Set random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True, enable cudnn determinism (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
