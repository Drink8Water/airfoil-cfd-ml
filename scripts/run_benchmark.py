#!/usr/bin/env python3
"""Benchmark runner: sweep over config files, train + evaluate each.

Usage:
  python scripts/run_benchmark.py --config-dir configs/experiment/benchmark_smoke
  python scripts/run_benchmark.py --config-dir configs/experiment/benchmark_smoke --cpu
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

import torch
import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v3 benchmark runner")
    p.add_argument(
        "--config-dir",
        required=True,
        help="Directory of YAML experiment configs to run.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU training.")
    p.add_argument(
        "--output-csv",
        default="reports/main_benchmark_smoke.csv",
        help="Output CSV path (default: reports/main_benchmark_smoke.csv).",
    )
    return p


def _train_one(config_path: str, prefer_cuda: bool) -> dict:
    """Train one experiment and return its extra_config dict + best checkpoint path."""
    from airfoil_cfd_ml.data import AirfoilNPZDataset, collate_stack, get_input_channels
    from airfoil_cfd_ml.models import build_model
    from airfoil_cfd_ml.losses.field import FieldLoss
    from airfoil_cfd_ml.losses.composite import CompositeLoss
    from airfoil_cfd_ml.training import Trainer, TrainerConfig
    from airfoil_cfd_ml.utils.device import resolve_device
    from torch.utils.data import DataLoader

    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Data
    data_cfg = cfg.get("data", {})
    train_dir = data_cfg.get("train_dir", "../train2")
    val_dir = data_cfg.get("val_dir", "../test")
    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 0))
    geometry_mode = data_cfg.get("geometry_mode", "mask_only")

    train_ds = AirfoilNPZDataset(train_dir, geometry_mode=geometry_mode)
    val_ds = AirfoilNPZDataset(val_dir, geometry_mode=geometry_mode)

    dev = resolve_device(prefer_cuda)
    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=collate_stack,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # Model
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "simple_cnn")
    model_kwargs = dict(model_cfg.get("kwargs", {}))
    if "in_channels" not in model_kwargs:
        model_kwargs["in_channels"] = get_input_channels(geometry_mode)
    model = build_model(model_name, **model_kwargs)

    # Loss
    loss_cfg = cfg.get("loss", {})
    loss_name = loss_cfg.get("name", "field")
    loss_kwargs = dict(loss_cfg.get("kwargs", {}))
    if loss_name == "field":
        loss_fn = FieldLoss(**loss_kwargs)
    elif loss_name == "composite":
        sub_losses = []
        for entry in loss_cfg.get("losses", []):
            sub_losses.append((FieldLoss(**entry.get("kwargs", {})), float(entry.get("weight", 1.0))))
        loss_fn = CompositeLoss(sub_losses)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    # Trainer
    train_cfg = cfg.get("train", {})
    experiment_name = config_path.stem
    save_dir = train_cfg.get("save_dir", f"outputs/checkpoints/{experiment_name}")

    trainer_config = TrainerConfig(
        epochs=int(train_cfg.get("epochs", 1)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        seed=int(train_cfg.get("seed", 42)),
        save_dir=save_dir,
    )

    extra_config = {
        "experiment_name": experiment_name,
        "config_file": str(config_path.resolve()),
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "geometry_mode": geometry_mode,
        "batch_size": batch_size,
    }

    trainer = Trainer(model, loss_fn, trainer_config, extra_config=extra_config, device=dev)
    trainer.fit(train_loader, val_loader)

    return {
        "save_dir": save_dir,
        "best_pt": str(Path(save_dir) / "best.pt"),
        "extra_config": extra_config,
    }


def _evaluate_one(best_pt: str, geometry_mode: str, val_dir: str, output_dir: str, prefer_cuda: bool) -> dict:
    """Evaluate a trained checkpoint and return the metrics dict."""
    from airfoil_cfd_ml.data import AirfoilNPZDataset, collate_stack, get_input_channels
    from airfoil_cfd_ml.models import build_model
    from airfoil_cfd_ml.evaluation.evaluator import Evaluator
    from airfoil_cfd_ml.utils.device import resolve_device
    from torch.utils.data import DataLoader
    import torch

    dev = resolve_device(prefer_cuda)

    # Infer model from checkpoint
    ckpt_data = torch.load(best_pt, map_location=dev, weights_only=False)
    extra = ckpt_data.get("extra_config", {})

    # Fallback: read from config_resolved.yaml (next to checkpoint)
    if not extra:
        resolved_yaml = Path(best_pt).parent / "config_resolved.yaml"
        if resolved_yaml.exists():
            with open(resolved_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            extra = cfg

    model_name = extra.get("model_name", "simple_cnn")
    model_kwargs = extra.get("model_kwargs", {})
    geometry_mode = extra.get("geometry_mode", geometry_mode)

    model = build_model(model_name, **model_kwargs)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(dev)

    test_ds = AirfoilNPZDataset(val_dir, geometry_mode=geometry_mode)
    test_loader = DataLoader(
        test_ds,
        batch_size=20,
        shuffle=False,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
        collate_fn=collate_stack,
    )

    evaluator = Evaluator(model, device=dev)
    return evaluator.evaluate(test_loader, output_dir=output_dir)


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        raise NotADirectoryError(str(config_dir))

    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        print(f"No YAML configs found in {config_dir}")
        return

    print(f"Found {len(config_files)} config(s) in {config_dir}")
    prefer_cuda = not args.cpu

    rows: list[dict] = []

    for cf in config_files:
        exp_name = cf.stem
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"Config: {cf}")

        try:
            # 1. Train
            train_result = _train_one(str(cf), prefer_cuda)
            save_dir = train_result["save_dir"]
            best_pt = train_result["best_pt"]
            extra = train_result["extra_config"]

            # 2. Evaluate
            # Use the val_dir from config for evaluation
            with open(cf, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            val_dir = cfg.get("data", {}).get("val_dir", "../test")
            eval_output_dir = os.path.join(save_dir, "eval")
            metrics = _evaluate_one(best_pt, extra["geometry_mode"], val_dir, eval_output_dir, prefer_cuda)

            row = {
                "experiment": exp_name,
                "model_name": extra.get("model_name", "?"),
                "geometry_mode": extra.get("geometry_mode", "?"),
                "device": metrics.get("device", "?"),
                "device_name": metrics.get("device_name", "?"),
                "input_channels": (
                    metrics.get("input_shape", [-1, -1, -1, -1])[1]
                    if isinstance(metrics.get("input_shape"), list)
                    and len(metrics["input_shape"]) >= 2
                    else "?"
                ),
                "input_resolution": (
                    metrics.get("input_shape", [-1, -1, -1, -1])[2:]
                    if isinstance(metrics.get("input_shape"), list)
                    and len(metrics["input_shape"]) >= 4
                    else "?"
                ),
                "status": "completed",
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            rows.append(row)
            print(f"  mean_rel_mae={metrics.get('mean_rel_mae', -1):.6f}")

        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            rows.append({
                "experiment": exp_name,
                "model_name": "?",
                "geometry_mode": "?",
                "status": "failed",
                "error": str(exc)[:200],
            })

    # Write CSV
    if rows:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Collect all fieldnames
        fieldnames = list(rows[0].keys())
        for r in rows[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nBenchmark CSV → {out_path}")
        n_ok = sum(1 for r in rows if r.get("status") == "completed")
        print(f"  {n_ok}/{len(rows)} completed")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()
