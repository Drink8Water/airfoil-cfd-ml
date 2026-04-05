from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from .data import TurbDataset
from .metrics import evaluate_loader_metrics
from .model import DfpNet
from .normalization import (
    NormalizationStats,
    NormalizedDataset,
    denormalize_from_neg1_pos1,
    normalize_target_with_pressure_scaling,
)


def _build_residual_baseline_targets_norm(
    inputs_norm: torch.Tensor,
    input_min: torch.Tensor,
    input_max: torch.Tensor,
    target_min: torch.Tensor,
    target_max: torch.Tensor,
) -> torch.Tensor:
    non_mask_phys = denormalize_from_neg1_pos1(inputs_norm[:, :2, :, :], input_min, input_max)
    fluid_mask = (inputs_norm[:, 2:3, :, :] < 0.5).float()

    base_pressure = torch.zeros_like(fluid_mask)
    base_velocity = non_mask_phys * fluid_mask
    base_target_phys = torch.cat([base_pressure, base_velocity], dim=1)
    return normalize_target_with_pressure_scaling(base_target_phys, target_min, target_max)


def evaluate_checkpoint(
    checkpoint_path: str,
    test_dir: str,
    batch_size: int = 20,
    num_workers: int = 0,
    prefer_cuda: bool = True,
) -> Dict[str, float]:
    device = torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    cfg = ckpt["config"]
    stats_raw = ckpt["stats"]
    stats = NormalizationStats(
        input_min=stats_raw["input_min"],
        input_max=stats_raw["input_max"],
        target_min=stats_raw["target_min"],
        target_max=stats_raw["target_max"],
    )

    model = DfpNet(channel_exponent=cfg.get("channel_exponent", 6), dropout=cfg.get("dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    use_residual_learning = bool(cfg.get("use_residual_learning", False))

    input_min = stats.input_min.to(device)
    input_max = stats.input_max.to(device)
    target_min = stats.target_min.to(device)
    target_max = stats.target_max.to(device)

    test_dataset = TurbDataset(test_dir)
    test_dataset_n = NormalizedDataset(test_dataset, stats)
    test_loader_n = torch.utils.data.DataLoader(
        test_dataset_n,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    return evaluate_loader_metrics(
        model=model,
        loader=test_loader_n,
        device=device,
        target_min=target_min,
        target_max=target_max,
        output_transform=(
            lambda inputs_norm, outputs_norm: outputs_norm
            + _build_residual_baseline_targets_norm(
                inputs_norm=inputs_norm,
                input_min=input_min,
                input_max=input_max,
                target_min=target_min,
                target_max=target_max,
            )
        )
        if use_residual_learning
        else None,
    )
