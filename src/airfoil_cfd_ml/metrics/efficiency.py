"""Efficiency metrics: parameter count and inference latency benchmarking."""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch module.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
    device: torch.device | None = None,
    warmup: int = 5,
    repeat: int = 20,
) -> dict:
    """Measure inference latency of a model.

    Args:
        model: PyTorch module (should already be on the target device).
        input_shape: (B, C, H, W) shape of the dummy input.
        device: Target device (default: model's current device).
        warmup: Number of warmup forward passes (excluded from timing).
        repeat: Number of timed forward passes.

    Returns:
        Dict with keys:
          mean_ms, std_ms, min_ms, max_ms, warmup, repeat,
          device, device_name, input_shape.
    """
    if device is None:
        device = next(model.parameters()).device

    # Resolve human-readable device name
    device_name = str(device)
    if device.type == "cuda":
        try:
            device_name = torch.cuda.get_device_name(device)
        except Exception:
            pass

    model.eval()
    dummy = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Timed
    times: list[float] = []
    with torch.no_grad():
        for _ in range(repeat):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

    t = torch.tensor(times, dtype=torch.float32)
    return {
        "mean_ms": round(float(t.mean().item()), 4),
        "std_ms": round(float(t.std().item()), 4),
        "min_ms": round(float(t.min().item()), 4),
        "max_ms": round(float(t.max().item()), 4),
        "warmup": warmup,
        "repeat": repeat,
        "device": str(device),
        "device_name": device_name,
        "input_shape": list(input_shape),
    }
