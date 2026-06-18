#!/usr/bin/env python3
"""v3 research framework evaluation CLI.

Two modes:
  1. --checkpoint + --config    → reconstruct model from experiment config.
  2. --checkpoint + --test-dir  → minimal: infer model from checkpoint extra_config.

Produces: eval_metrics.json, per_sample_metrics.csv

Usage:
  python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke/best.pt --config configs/experiment/smoke_simple_cnn_mask_only.yaml
  python scripts/evaluate.py --checkpoint outputs/checkpoints/smoke/best.pt --test-dir ../test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from airfoil_cfd_ml.data import AirfoilNPZDataset, collate_stack, get_input_channels
from airfoil_cfd_ml.models import build_model
from airfoil_cfd_ml.evaluation.evaluator import Evaluator
from airfoil_cfd_ml.utils.device import resolve_device


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v3 research framework: evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--config", default=None, help="Path to YAML experiment config.")
    p.add_argument("--test-dir", default=None, help="Path to test .npz directory (overrides config val_dir).")
    p.add_argument("--output-dir", default=None, help="Output directory (default: <checkpoint_dir>/eval).")
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    p.add_argument("--batch-size", type=int, default=20, help="Batch size.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    return p


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = args.output_dir or str(ckpt_path.parent / "eval")

    # ------------------------------------------------------------------
    # Determine model config
    # ------------------------------------------------------------------
    geometry_mode = "mask_only"
    model_name = None
    model_kwargs: dict = {}

    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get("data", {})
        geometry_mode = data_cfg.get("geometry_mode", "mask_only")

        model_cfg = cfg.get("model", {})
        model_name = model_cfg.get("name", "simple_cnn")
        model_kwargs = dict(model_cfg.get("kwargs", {}))
        if "in_channels" not in model_kwargs:
            model_kwargs["in_channels"] = get_input_channels(geometry_mode)

    # Fallback: infer from checkpoint
    if model_name is None:
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        extra = ckpt_data.get("extra_config", {}) if isinstance(ckpt_data, dict) else {}
        model_name = extra.get("model_name", "simple_cnn")
        model_kwargs = extra.get("model_kwargs", {})
        geometry_mode = extra.get("geometry_mode", "mask_only")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    test_dir = args.test_dir
    if test_dir is None and args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            test_dir = yaml.safe_load(f).get("data", {}).get("val_dir", "../test")
    if test_dir is None:
        test_dir = "../test"

    dev = resolve_device(prefer_cuda=not args.cpu)
    test_ds = AirfoilNPZDataset(test_dir, geometry_mode=geometry_mode)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dev.type == "cuda"),
        collate_fn=collate_stack,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(model_name, **model_kwargs)
    ckpt_data = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(dev)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print(f"Checkpoint: {ckpt_path}")
    print(f"  model: {model_name}, geometry: {geometry_mode}, input_C: {model_kwargs.get('in_channels', '?')}")
    print(f"  test_dir: {test_dir}, samples: {len(test_ds)}")

    evaluator = Evaluator(model, device=dev)
    metrics = evaluator.evaluate(test_loader, output_dir=output_dir)

    print(f"\nAggregate metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.6f}")
    print(f"\nReports → {output_dir}")


if __name__ == "__main__":
    main()
