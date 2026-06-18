"""Tests for efficiency metrics: count_parameters, benchmark_latency.

All tests use synthetic models — no real data required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from airfoil_cfd_ml.metrics.efficiency import benchmark_latency, count_parameters


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestCountParameters:
    def test_positive(self):
        model = _TinyModel()
        n = count_parameters(model)
        assert n > 0
        assert isinstance(n, int)

    def test_trainable_only(self):
        model = _TinyModel()
        n_all = count_parameters(model, trainable_only=False)
        n_train = count_parameters(model, trainable_only=True)
        assert n_train == n_all  # all params trainable in this model


class TestBenchmarkLatency:
    def test_returns_dict(self):
        model = _TinyModel()
        lat = benchmark_latency(model, input_shape=(1, 3, 32, 32), warmup=2, repeat=5)
        assert isinstance(lat, dict)
        for key in ("mean_ms", "std_ms", "min_ms", "max_ms", "device"):
            assert key in lat

    def test_cpu_works(self):
        model = _TinyModel()
        lat = benchmark_latency(
            model, input_shape=(1, 3, 16, 16),
            device=torch.device("cpu"), warmup=1, repeat=3,
        )
        assert lat["mean_ms"] > 0.0

    def test_gpu_if_available(self):
        if not torch.cuda.is_available():
            return
        model = _TinyModel().cuda()
        lat = benchmark_latency(
            model, input_shape=(1, 3, 16, 16),
            device=torch.device("cuda"), warmup=2, repeat=5,
        )
        assert lat["mean_ms"] > 0.0
        assert lat["device"] == "cuda"
