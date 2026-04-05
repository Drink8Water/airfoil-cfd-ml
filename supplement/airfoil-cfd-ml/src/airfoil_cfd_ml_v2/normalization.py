from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_to_neg1_pos1(tensor: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor, eps: float = 1e-8):
    min_val = min_val.to(tensor.device)
    max_val = max_val.to(tensor.device)
    range_val = max_val - min_val
    range_val = torch.where(range_val < eps, tensor.new_tensor(eps), range_val)
    return 2 * (tensor - min_val) / range_val - 1


def denormalize_from_neg1_pos1(normalized: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
    min_val = min_val.to(normalized.device)
    max_val = max_val.to(normalized.device)
    return ((normalized + 1) / 2) * (max_val - min_val) + min_val


def normalize_input(inputs: torch.Tensor, input_min: torch.Tensor, input_max: torch.Tensor):
    non_mask = inputs[:, :2, :, :]
    mask = inputs[:, 2:, :, :]
    normalized_non_mask = normalize_to_neg1_pos1(non_mask, input_min, input_max)
    return torch.cat([normalized_non_mask, mask], dim=1)


def normalize_target_with_pressure_scaling(targets: torch.Tensor, target_min: torch.Tensor, target_max: torch.Tensor):
    pressure = targets[:, 0:1]
    velocity = targets[:, 1:]

    pressure_log = torch.sign(pressure) * torch.log1p(torch.abs(pressure))
    pressure_norm = normalize_to_neg1_pos1(pressure_log, target_min[0:1], target_max[0:1])
    velocity_norm = normalize_to_neg1_pos1(velocity, target_min[1:], target_max[1:])

    return torch.cat([pressure_norm, velocity_norm], dim=1)


def denormalize_target_with_pressure_scaling(
    normalized_target: torch.Tensor,
    target_min: torch.Tensor,
    target_max: torch.Tensor,
    clamp_normalized: bool = False,
):
    if clamp_normalized:
        normalized_target = normalized_target.clamp(-1.0, 1.0)

    pressure_norm = normalized_target[:, 0:1]
    velocity_norm = normalized_target[:, 1:]

    pressure_log = denormalize_from_neg1_pos1(pressure_norm, target_min[0:1], target_max[0:1])
    pressure = torch.sign(pressure_log) * torch.expm1(torch.abs(pressure_log))

    velocity = denormalize_from_neg1_pos1(velocity_norm, target_min[1:], target_max[1:])
    return torch.cat([pressure, velocity], dim=1)


def compute_global_stats(dataset):
    input_mins = [[] for _ in range(2)]
    input_maxs = [[] for _ in range(2)]
    target_mins = [[] for _ in range(3)]
    target_maxs = [[] for _ in range(3)]

    for inputs, targets in dataset:
        non_mask_input = inputs[:2]
        for channel in range(2):
            input_mins[channel].append(non_mask_input[channel].min().item())
            input_maxs[channel].append(non_mask_input[channel].max().item())

        pressure = targets[0]
        pressure_log = torch.sign(pressure) * torch.log1p(torch.abs(pressure))
        target_mins[0].append(pressure_log.min().item())
        target_maxs[0].append(pressure_log.max().item())

        for channel in [1, 2]:
            target_mins[channel].append(targets[channel].min().item())
            target_maxs[channel].append(targets[channel].max().item())

    return {
        "input_min": torch.tensor([np.min(channel) for channel in input_mins], dtype=torch.float32).view(2, 1, 1),
        "input_max": torch.tensor([np.max(channel) for channel in input_maxs], dtype=torch.float32).view(2, 1, 1),
        "target_min": torch.tensor([np.min(channel) for channel in target_mins], dtype=torch.float32).view(3, 1, 1),
        "target_max": torch.tensor([np.max(channel) for channel in target_maxs], dtype=torch.float32).view(3, 1, 1),
    }


@dataclass
class NormalizationStats:
    input_min: torch.Tensor
    input_max: torch.Tensor
    target_min: torch.Tensor
    target_max: torch.Tensor


class NormalizedDataset(Dataset):
    def __init__(self, raw_dataset, stats: NormalizationStats):
        self.raw_dataset = raw_dataset
        self.stats = stats

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        inputs, targets = self.raw_dataset[idx]
        inputs = inputs.float()
        targets = targets.float()

        inputs_norm = normalize_input(
            inputs.unsqueeze(0),
            self.stats.input_min,
            self.stats.input_max,
        ).squeeze(0)
        targets_norm = normalize_target_with_pressure_scaling(
            targets.unsqueeze(0),
            self.stats.target_min,
            self.stats.target_max,
        ).squeeze(0)
        return inputs_norm, targets_norm
