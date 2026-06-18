"""Model registry: name → constructor mapping, config-driven build, parameter count."""

from __future__ import annotations

from typing import Any, Dict, Type

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator: register a model class under a string name.

    Usage:
        @register_model("my_model")
        class MyModel(nn.Module):
            ...
    """

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' already registered.")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered model by name.

    Args:
        name: registry key (e.g. "simple_cnn").
        **kwargs: forwarded to the model constructor.

    Returns:
        Instantiated nn.Module.

    Raises:
        KeyError: if the name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def build_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """Build a model from a flat configuration dictionary.

    The config dict MUST contain at least ``model_name``.  All other keys
    are forwarded to the model constructor after stripping the ``model_``
    prefix (so ``model_width=32`` becomes ``width=32``).

    Args:
        config: dict with keys like ``model_name``, ``model_width``, etc.

    Returns:
        Instantiated nn.Module.
    """
    config = dict(config)
    name = config.pop("model_name", None)
    if name is None:
        raise ValueError("config must contain 'model_name' key")

    # Strip "model_" prefix from kwargs
    kwargs: Dict[str, Any] = {}
    for k, v in config.items():
        if k.startswith("model_"):
            kwargs[k[len("model_"):]] = v
        else:
            kwargs[k] = v

    return build_model(name, **kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of (trainable) parameters in a model.

    Args:
        model: PyTorch module.
        trainable_only: if True, only count parameters with requires_grad=True.

    Returns:
        Total parameter count (int).
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_millions(model: nn.Module, trainable_only: bool = True) -> float:
    """Convenience: count_parameters / 1e6, rounded to 2 decimal places."""
    return round(count_parameters(model, trainable_only) / 1e6, 2)
