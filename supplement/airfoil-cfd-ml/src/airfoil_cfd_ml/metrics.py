from __future__ import annotations

from typing import Dict

import torch

from .normalization import denormalize_target_with_pressure_scaling


def evaluate_loader_metrics(
    model,
    loader,
    device: torch.device,
    target_min: torch.Tensor,
    target_max: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    mae_sum = torch.zeros(3, device=device)
    mse_sum = torch.zeros(3, device=device)
    tgt_abs_sum = torch.zeros(3, device=device)
    valid_points = 0.0

    with torch.no_grad():
        for inputs, targets_norm in loader:
            inputs = inputs.to(device)
            targets_norm = targets_norm.to(device)

            outputs_norm = model(inputs).clamp(-1.0, 1.0)
            targets_phys = denormalize_target_with_pressure_scaling(
                targets_norm,
                target_min,
                target_max,
                clamp_normalized=True,
            )
            outputs_phys = denormalize_target_with_pressure_scaling(
                outputs_norm,
                target_min,
                target_max,
                clamp_normalized=True,
            )

            fluid_mask = (inputs[:, 2:3, :, :] < 0.5).float()
            err = outputs_phys - targets_phys
            abs_err = err.abs() * fluid_mask
            sq_err = err.pow(2) * fluid_mask
            abs_tgt = targets_phys.abs() * fluid_mask

            mae_sum += abs_err.sum(dim=(0, 2, 3))
            mse_sum += sq_err.sum(dim=(0, 2, 3))
            tgt_abs_sum += abs_tgt.sum(dim=(0, 2, 3))
            valid_points += fluid_mask.sum().item()

    mae = mae_sum / max(valid_points, 1.0)
    rmse = torch.sqrt(mse_sum / max(valid_points, 1.0))
    rel_mae = mae_sum / (tgt_abs_sum + 1e-8)

    return {
        "pressure_mae": mae[0].item(),
        "u_mae": mae[1].item(),
        "v_mae": mae[2].item(),
        "pressure_rmse": rmse[0].item(),
        "u_rmse": rmse[1].item(),
        "v_rmse": rmse[2].item(),
        "pressure_rel_mae": rel_mae[0].item(),
        "u_rel_mae": rel_mae[1].item(),
        "v_rel_mae": rel_mae[2].item(),
    }
