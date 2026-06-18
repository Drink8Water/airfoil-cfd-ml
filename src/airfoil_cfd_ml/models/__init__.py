"""Model zoo: registry, base class, and all model architectures.

Importing this package auto-registers all models via @register_model decorators.
"""

from .registry import (
    MODEL_REGISTRY,
    build_model,
    build_model_from_config,
    count_parameters,
    count_parameters_millions,
    register_model,
)
from .base import BaseModel
from .simple_cnn import SimpleCNN
from .res_unet import ResUNet
from .fno2d import FNO2D, SpectralConv2d
from .geofno_lite import GeoFNOLite
from .transolver_lite import TransolverLite

__all__ = [
    # Registry
    "MODEL_REGISTRY",
    "register_model",
    "build_model",
    "build_model_from_config",
    "count_parameters",
    "count_parameters_millions",
    # Base
    "BaseModel",
    # Models
    "SimpleCNN",
    "ResUNet",
    "FNO2D",
    "SpectralConv2d",
    "GeoFNOLite",
    "TransolverLite",
]
