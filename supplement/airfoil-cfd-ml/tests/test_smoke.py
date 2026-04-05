import torch

from airfoil_cfd_ml.model import DfpNet
from airfoil_cfd_ml.normalization import (
    normalize_input,
    normalize_target_with_pressure_scaling,
    denormalize_target_with_pressure_scaling,
)


def test_forward_shape():
    model = DfpNet(channel_exponent=4)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 3, 128, 128)


def test_norm_denorm_pipeline():
    inputs = torch.randn(2, 3, 128, 128)
    inputs[:, 2:3] = (inputs[:, 2:3] > 0).float()
    targets = torch.randn(2, 3, 128, 128) * 100.0

    input_min = torch.tensor([-1.0, -1.0]).view(2, 1, 1)
    input_max = torch.tensor([1.0, 1.0]).view(2, 1, 1)
    target_min = torch.tensor([-5.0, -200.0, -200.0]).view(3, 1, 1)
    target_max = torch.tensor([5.0, 200.0, 200.0]).view(3, 1, 1)

    x_norm = normalize_input(inputs, input_min, input_max)
    t_norm = normalize_target_with_pressure_scaling(targets, target_min, target_max)
    t_back = denormalize_target_with_pressure_scaling(t_norm, target_min, target_max)

    assert x_norm.shape == inputs.shape
    assert t_back.shape == targets.shape
