"""Checkpoint utilities: save and load model checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any = None,
    epoch: int | None = None,
    **extra,
) -> None:
    """Save a checkpoint to disk.

    Args:
        path: Output file path (e.g. ``best.pt``).
        model: The model to save.
        optimizer: Optional optimizer (state_dict saved).
        epoch: Optional epoch number.
        **extra: Additional data to store (config, history, metrics, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch
    data.update(extra)

    torch.save(data, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Any = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint from disk.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into (in-place).
        optimizer: Optional optimizer to load state into (in-place).
        map_location: Device string for torch.load.

    Returns:
        The full checkpoint dict (extra keys accessible by caller).
    """
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
