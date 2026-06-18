from .regression import compute_regression_metrics
from .physics import (
    boundary_rel_mae,
    divergence_error,
    finite_diff_x,
    finite_diff_y,
    vorticity_error,
    wake_rel_mae,
)
from .spectral import energy_spectrum_error, spectral_error
from .efficiency import benchmark_latency, count_parameters
from .aggregator import aggregate_metric_dicts

__all__ = [
    # Regression
    "compute_regression_metrics",
    # Physics
    "finite_diff_x",
    "finite_diff_y",
    "divergence_error",
    "vorticity_error",
    "boundary_rel_mae",
    "wake_rel_mae",
    # Spectral
    "spectral_error",
    "energy_spectrum_error",
    # Efficiency
    "count_parameters",
    "benchmark_latency",
    # Aggregation
    "aggregate_metric_dicts",
]
