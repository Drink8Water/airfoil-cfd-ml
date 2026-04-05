from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .data import create_splits_and_loaders
from .losses import CompositeV2Loss
from .metrics import evaluate_loader_metrics
from .model import DfpNet, weights_init
from .normalization import (
    NormalizationStats,
    NormalizedDataset,
    compute_global_stats,
    denormalize_from_neg1_pos1,
    normalize_target_with_pressure_scaling,
)


@dataclass
class TrainConfig:
    train_dir: str
    test_dir: str
    batch_size: int = 20
    num_workers: int = 0
    val_ratio: float = 0.1
    seed: int = 42
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-3
    betas: tuple = (0.5, 0.999)
    channel_exponent: int = 6
    dropout: float = 0.0
    loss_weights: tuple = (5.0, 1.0, 1.0)
    gradient_loss_weights: tuple = (1.0, 1.0, 1.0)
    grad_loss_weight: float = 0.2
    divergence_loss_weight: float = 0.0
    divergence_dvdy_scale: float = 1.0
    divergence_interior_margin: int = 0
    divergence_use_physical_space: bool = False
    divergence_warmup_ratio: float = 0.0
    adaptive_divergence: bool = False
    adaptive_divergence_init_weight: float = 0.0
    adaptive_divergence_min_weight: float = 0.0
    adaptive_divergence_max_weight: float = 0.0
    adaptive_divergence_ema: float = 0.9
    adaptive_divergence_up_factor: float = 1.05
    adaptive_divergence_down_factor: float = 0.7
    adaptive_divergence_pressure_rel_ref: float = 0.0
    adaptive_divergence_pressure_rel_tolerance: float = 0.02
    use_residual_learning: bool = False
    edge_taper_width: int = 4
    edge_taper_power: float = 2.0
    obstacle_edge_width: int = 2
    obstacle_edge_weight: float = 0.85
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4
    save_dir: str = "checkpoints"
    init_checkpoint: str | None = None


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_border_taper(height: int, width: int, device: torch.device, taper_width: int, taper_power: float) -> torch.Tensor:
    if taper_width <= 0:
        return torch.ones((1, 1, height, width), device=device)

    yy = torch.arange(height, device=device, dtype=torch.float32).view(height, 1).expand(height, width)
    xx = torch.arange(width, device=device, dtype=torch.float32).view(1, width).expand(height, width)

    dist_left = xx
    dist_right = (width - 1) - xx
    dist_top = yy
    dist_bottom = (height - 1) - yy
    dist_to_border = torch.minimum(torch.minimum(dist_left, dist_right), torch.minimum(dist_top, dist_bottom))

    taper = (dist_to_border / float(taper_width)).clamp(0.0, 1.0)
    taper = taper.pow(max(taper_power, 1e-6))
    return taper.view(1, 1, height, width)


def _build_spatial_weight(
    inputs: torch.Tensor,
    fluid_mask: torch.Tensor,
    border_taper: torch.Tensor,
    obstacle_edge_width: int,
    obstacle_edge_weight: float,
) -> torch.Tensor:
    spatial_weight = fluid_mask * border_taper

    if obstacle_edge_width <= 0:
        return spatial_weight

    kernel = 2 * obstacle_edge_width + 1
    solid = (inputs[:, 2:3, :, :] >= 0.5).float()
    solid_dilated = F.max_pool2d(solid, kernel_size=kernel, stride=1, padding=obstacle_edge_width)
    solid_eroded = 1.0 - F.max_pool2d(1.0 - solid, kernel_size=kernel, stride=1, padding=obstacle_edge_width)
    obstacle_ring = (solid_dilated - solid_eroded).clamp(0.0, 1.0)
    obstacle_ring_in_fluid = obstacle_ring * fluid_mask

    safe_edge_weight = float(min(max(obstacle_edge_weight, 0.0), 1.0))
    ring_weight = 1.0 - obstacle_ring_in_fluid * (1.0 - safe_edge_weight)
    return spatial_weight * ring_weight


def _build_residual_baseline_targets_norm(
    inputs_norm: torch.Tensor,
    input_min: torch.Tensor,
    input_max: torch.Tensor,
    target_min: torch.Tensor,
    target_max: torch.Tensor,
) -> torch.Tensor:
    # Baseline field: p=0, velocity=(u_inf_x, u_inf_y) in fluid region.
    non_mask_phys = denormalize_from_neg1_pos1(inputs_norm[:, :2, :, :], input_min, input_max)
    fluid_mask = (inputs_norm[:, 2:3, :, :] < 0.5).float()

    base_pressure = torch.zeros_like(fluid_mask)
    base_velocity = non_mask_phys * fluid_mask
    base_target_phys = torch.cat([base_pressure, base_velocity], dim=1)
    return normalize_target_with_pressure_scaling(base_target_phys, target_min, target_max)


def _scheduled_divergence_weight(epoch_idx: int, total_epochs: int, target_weight: float, warmup_ratio: float) -> float:
    target_weight = float(max(0.0, target_weight))
    if target_weight <= 0.0:
        return 0.0

    total_epochs = max(int(total_epochs), 1)
    warmup_ratio = float(min(max(warmup_ratio, 0.0), 0.999999))
    progress = float(epoch_idx + 1) / float(total_epochs)

    if progress <= warmup_ratio:
        return 0.0

    ramp = (progress - warmup_ratio) / max(1e-8, 1.0 - warmup_ratio)
    ramp = min(max(ramp, 0.0), 1.0)
    return target_weight * ramp


def run_training(config: TrainConfig, prefer_cuda: bool = True) -> Dict[str, float]:
    device = resolve_device(prefer_cuda)

    bundle = create_splits_and_loaders(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    stats_raw = compute_global_stats(bundle.train_dataset)
    stats = NormalizationStats(
        input_min=stats_raw["input_min"],
        input_max=stats_raw["input_max"],
        target_min=stats_raw["target_min"],
        target_max=stats_raw["target_max"],
    )
    stats_input_min = stats.input_min.to(device)
    stats_input_max = stats.input_max.to(device)
    stats_target_min = stats.target_min.to(device)
    stats_target_max = stats.target_max.to(device)

    train_dataset_n = NormalizedDataset(bundle.train_dataset, stats)
    val_dataset_n = NormalizedDataset(bundle.val_dataset, stats)
    test_dataset_n = NormalizedDataset(bundle.test_dataset, stats)

    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=(device.type == "cuda"))
    train_loader_n = torch.utils.data.DataLoader(train_dataset_n, shuffle=True, **loader_kwargs)
    val_loader_n = torch.utils.data.DataLoader(val_dataset_n, shuffle=False, **loader_kwargs)
    test_loader_n = torch.utils.data.DataLoader(test_dataset_n, shuffle=False, **loader_kwargs)

    model = DfpNet(channel_exponent=config.channel_exponent, dropout=config.dropout).to(device)
    model.apply(weights_init)

    if config.init_checkpoint:
        init_ckpt = torch.load(config.init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"])

    criterion = CompositeV2Loss(
        value_weights=config.loss_weights,
        gradient_weights=config.gradient_loss_weights,
        grad_loss_weight=config.grad_loss_weight,
        divergence_loss_weight=0.0,
        divergence_dvdy_scale=config.divergence_dvdy_scale,
        divergence_interior_margin=config.divergence_interior_margin,
        divergence_use_physical_space=config.divergence_use_physical_space,
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    loss_hist = []
    loss_hist_val = []
    epoch_records = []
    best_val = float("inf")
    epochs_without_improve = 0

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "dfpnet_best.pt"

    pbar = tqdm(range(config.epochs), ncols=100)
    adaptive_div_w = float(max(0.0, config.adaptive_divergence_init_weight))

    for epoch_idx in pbar:
        if config.adaptive_divergence:
            current_div_w = adaptive_div_w
        else:
            current_div_w = _scheduled_divergence_weight(
                epoch_idx=epoch_idx,
                total_epochs=config.epochs,
                target_weight=config.divergence_loss_weight,
                warmup_ratio=config.divergence_warmup_ratio,
            )
        criterion.divergence_loss_weight = current_div_w

        model.train()
        train_loss_acc = 0.0

        for inputs, targets in train_loader_n:
            inputs = inputs.to(device)
            targets = targets.to(device)
            fluid_mask = (inputs[:, 2:3, :, :] < 0.5).float()
            border_taper = _build_border_taper(
                height=inputs.shape[2],
                width=inputs.shape[3],
                device=inputs.device,
                taper_width=config.edge_taper_width,
                taper_power=config.edge_taper_power,
            )
            spatial_weight = _build_spatial_weight(
                inputs=inputs,
                fluid_mask=fluid_mask,
                border_taper=border_taper,
                obstacle_edge_width=config.obstacle_edge_width,
                obstacle_edge_weight=config.obstacle_edge_weight,
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            if config.use_residual_learning:
                baseline_norm = _build_residual_baseline_targets_norm(
                    inputs_norm=inputs,
                    input_min=stats_input_min,
                    input_max=stats_input_max,
                    target_min=stats_target_min,
                    target_max=stats_target_max,
                )
                outputs = outputs + baseline_norm
            loss = criterion(
                outputs,
                targets,
                spatial_weight=spatial_weight,
                target_min=stats_target_min,
                target_max=stats_target_max,
            )
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()

        train_loss = train_loss_acc / max(len(train_loader_n), 1)
        loss_hist.append(train_loss)

        model.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader_n:
                inputs = inputs.to(device)
                targets = targets.to(device)
                fluid_mask = (inputs[:, 2:3, :, :] < 0.5).float()
                border_taper = _build_border_taper(
                    height=inputs.shape[2],
                    width=inputs.shape[3],
                    device=inputs.device,
                    taper_width=config.edge_taper_width,
                    taper_power=config.edge_taper_power,
                )
                spatial_weight = _build_spatial_weight(
                    inputs=inputs,
                    fluid_mask=fluid_mask,
                    border_taper=border_taper,
                    obstacle_edge_width=config.obstacle_edge_width,
                    obstacle_edge_weight=config.obstacle_edge_weight,
                )
                outputs = model(inputs)
                if config.use_residual_learning:
                    baseline_norm = _build_residual_baseline_targets_norm(
                        inputs_norm=inputs,
                        input_min=stats_input_min,
                        input_max=stats_input_max,
                        target_min=stats_target_min,
                        target_max=stats_target_max,
                    )
                    outputs = outputs + baseline_norm
                val_loss_acc += criterion(
                    outputs,
                    targets,
                    spatial_weight=spatial_weight,
                    target_min=stats_target_min,
                    target_max=stats_target_max,
                ).item()

        val_loss = val_loss_acc / max(len(val_loader_n), 1)
        loss_hist_val.append(val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if config.adaptive_divergence:
            val_metrics = evaluate_loader_metrics(
                model=model,
                loader=val_loader_n,
                device=device,
                target_min=stats_target_min,
                target_max=stats_target_max,
                output_transform=(
                    lambda inputs_norm, outputs_norm: outputs_norm
                    + _build_residual_baseline_targets_norm(
                        inputs_norm=inputs_norm,
                        input_min=stats_input_min,
                        input_max=stats_input_max,
                        target_min=stats_target_min,
                        target_max=stats_target_max,
                    )
                )
                if config.use_residual_learning
                else None,
            )

            pressure_rel = float(val_metrics.get("pressure_rel_mae", 0.0))
            ref = float(max(0.0, config.adaptive_divergence_pressure_rel_ref))
            tol = float(max(0.0, config.adaptive_divergence_pressure_rel_tolerance))
            up_factor = float(max(1.0, config.adaptive_divergence_up_factor))
            down_factor = float(min(max(config.adaptive_divergence_down_factor, 0.0), 1.0))
            ema = float(min(max(config.adaptive_divergence_ema, 0.0), 0.999999))

            if ref > 0.0 and pressure_rel > ref * (1.0 + tol):
                target_w = adaptive_div_w * down_factor
            else:
                target_w = adaptive_div_w * up_factor

            w_min = float(max(0.0, config.adaptive_divergence_min_weight))
            w_max = float(max(w_min, config.adaptive_divergence_max_weight))
            if w_max > 0.0:
                target_w = min(max(target_w, w_min), w_max)

            adaptive_div_w = ema * adaptive_div_w + (1.0 - ema) * target_w
            adaptive_div_w = float(max(0.0, adaptive_div_w))

        if val_loss < (best_val - config.early_stopping_min_delta):
            best_val = val_loss
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stats": {
                        "input_min": stats.input_min,
                        "input_max": stats.input_max,
                        "target_min": stats.target_min,
                        "target_max": stats.target_max,
                    },
                    "config": config.__dict__,
                    "best_val_loss": best_val,
                },
                best_path,
            )
        else:
            epochs_without_improve += 1

        epoch_records.append(
            {
                "epoch": epoch_idx + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "divergence_loss_weight": current_div_w,
                "best_val_loss_so_far": best_val,
            }
        )

        pbar.set_description(
            f"train={train_loss:.5f} val={val_loss:.5f} div={current_div_w:.4g} wait={epochs_without_improve}/{config.early_stopping_patience}"
        )

        if config.early_stopping_patience > 0 and epochs_without_improve >= config.early_stopping_patience:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_loader_metrics(
        model=model,
        loader=test_loader_n,
        device=device,
        target_min=stats_target_min,
        target_max=stats_target_max,
        output_transform=(
            lambda inputs_norm, outputs_norm: outputs_norm
            + _build_residual_baseline_targets_norm(
                inputs_norm=inputs_norm,
                input_min=stats_input_min,
                input_max=stats_input_max,
                target_min=stats_target_min,
                target_max=stats_target_max,
            )
        )
        if config.use_residual_learning
        else None,
    )

    np.save(save_dir / "loss_train.npy", np.asarray(loss_hist))
    np.save(save_dir / "loss_val.npy", np.asarray(loss_hist_val))

    metrics_csv_path = save_dir / "epoch_metrics.csv"
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "val_loss", "lr", "divergence_loss_weight", "best_val_loss_so_far"],
        )
        writer.writeheader()
        writer.writerows(epoch_records)

    curve_path = save_dir / "training_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(loss_hist) + 1), loss_hist, label="train_loss")
    plt.plot(np.arange(1, len(loss_hist_val) + 1), loss_hist_val, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()

    return {
        "best_val_loss": best_val,
        **metrics,
    }
