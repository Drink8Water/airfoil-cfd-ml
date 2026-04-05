from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from airfoil_cfd_ml.train import TrainConfig, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    config = TrainConfig(**cfg)
    metrics = run_training(config, prefer_cuda=not args.cpu)

    print("=== Training Done ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
