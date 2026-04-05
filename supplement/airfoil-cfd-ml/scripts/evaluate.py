from __future__ import annotations

import argparse

from airfoil_cfd_ml.evaluate import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefer_cuda=not args.cpu,
    )

    print("=== Test Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
