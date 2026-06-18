#!/usr/bin/env python3
"""v3 research framework training CLI.

Usage:
  python scripts/train.py --config configs/experiment/smoke_simple_cnn_mask_only.yaml
  python scripts/train.py --config configs/experiment/smoke_simple_cnn_mask_only.yaml --cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Allow running from repo root without install
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from airfoil_cfd_ml.data import AirfoilNPZDataset, collate_stack, get_input_channels
from airfoil_cfd_ml.models import build_model
from airfoil_cfd_ml.losses.field import FieldLoss
from airfoil_cfd_ml.losses.composite import CompositeLoss
from airfoil_cfd_ml.training import Trainer, TrainerConfig
from airfoil_cfd_ml.utils.device import resolve_device


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v3 research framework: training")
    p.add_argument("--config", required=True, help="Path to YAML experiment config.")
    p.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return p


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    data_cfg = cfg.get("data", {})
    train_dir = data_cfg.get("train_dir", "../train2")
    val_dir = data_cfg.get("val_dir", "../test")
    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 0))
    geometry_mode = data_cfg.get("geometry_mode", "mask_only")

    train_ds = AirfoilNPZDataset(train_dir, geometry_mode=geometry_mode)
    val_ds = AirfoilNPZDataset(val_dir, geometry_mode=geometry_mode)

    dev = resolve_device(prefer_cuda=not args.cpu)
    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=collate_stack,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "simple_cnn")
    model_kwargs = dict(model_cfg.get("kwargs", {}))
    # Auto-derive in_channels from geometry_mode
    if "in_channels" not in model_kwargs:
        model_kwargs["in_channels"] = get_input_channels(geometry_mode)

    model = build_model(model_name, **model_kwargs)

    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    # ------------------------------------------------------------------
    # 3. Loss
    # ------------------------------------------------------------------
    loss_cfg = cfg.get("loss", {})
    loss_name = loss_cfg.get("name", "field")
    loss_kwargs = dict(loss_cfg.get("kwargs", {}))

    if loss_name == "field":
        loss_fn = FieldLoss(**loss_kwargs)
    elif loss_name == "composite":
        sub_losses = []
        for entry in loss_cfg.get("losses", []):
            sub_name = entry.get("name", "field")
            sub_kw = dict(entry.get("kwargs", {}))
            sub_weight = float(entry.get("weight", 1.0))
            if sub_name == "field":
                sub_losses.append((FieldLoss(**sub_kw), sub_weight))
            else:
                raise ValueError(f"Unknown sub-loss name: {sub_name}")
        loss_fn = CompositeLoss(sub_losses)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")

    # ------------------------------------------------------------------
    # 4. Trainer
    # ------------------------------------------------------------------
    train_cfg = cfg.get("train", {})
    experiment_name = config_path.stem
    save_dir = train_cfg.get("save_dir", f"outputs/checkpoints/{experiment_name}")

    trainer_config = TrainerConfig(
        epochs=int(train_cfg.get("epochs", 50)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        seed=int(train_cfg.get("seed", 42)),
        save_dir=save_dir,
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(train_cfg.get("early_stopping_min_delta", 1e-4)),
    )

    extra_config = {
        "experiment_name": experiment_name,
        "config_file": str(config_path.resolve()),
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "geometry_mode": geometry_mode,
        "train_dir": str(Path(train_dir).resolve()),
        "val_dir": str(Path(val_dir).resolve()),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "loss_name": loss_name,
        "loss_kwargs": loss_kwargs,
        "dataset_size": len(train_ds),
        "val_dataset_size": len(val_ds),
    }

    trainer = Trainer(model, loss_fn, trainer_config, extra_config=extra_config, device=dev)

    # ------------------------------------------------------------------
    # 5. Run
    # ------------------------------------------------------------------
    print(f"Experiment: {experiment_name}")
    print(f"  model: {model_name}, geometry: {geometry_mode}, input_C: {model_kwargs['in_channels']}")
    print(f"  train samples: {len(train_ds)}, val samples: {len(val_ds)}")
    print(f"  epochs: {trainer_config.epochs}, lr: {trainer_config.lr}, batch: {batch_size}")
    print(f"  save_dir: {save_dir}")
    print()

    history = trainer.fit(train_loader, val_loader)

    print(f"\nTraining complete: {len(history)} epochs, outputs → {save_dir}")


if __name__ == "__main__":
    main()
