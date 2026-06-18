"""airfoil-cfd-ml: Research-oriented SciML framework for 2D airfoil CFD surrogate modeling.

Layered architecture:
  data/         — datasets, schemas, transforms
  models/       — model registry, base classes, architectures
  losses/       — field losses, composite losses
  metrics/      — regression metrics
  training/     — trainer, callbacks
  evaluation/   — evaluator
  visualization/— plotting utilities
  utils/        — device resolution, helpers
"""

__version__ = "0.2.0"
