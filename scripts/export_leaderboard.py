#!/usr/bin/env python3
"""Export leaderboard: sort a benchmark CSV and (optionally) plot a bar chart.

Usage:
  python scripts/export_leaderboard.py --input reports/main_benchmark_smoke.csv
  python scripts/export_leaderboard.py --input reports/main_benchmark_smoke.csv --plot
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export sorted leaderboard from benchmark CSV.")
    p.add_argument("--input", required=True, help="Path to benchmark CSV.")
    p.add_argument(
        "--output",
        default=None,
        help="Output sorted CSV path (default: <input>_sorted.csv).",
    )
    p.add_argument(
        "--sort-by",
        default="mean_rel_mae",
        help="Column to sort by (ascending, default: mean_rel_mae).",
    )
    p.add_argument("--plot", action="store_true", help="Generate a matplotlib bar chart.")
    return p


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    # Read
    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows found.")
        return

    # Sort
    sort_key = args.sort_by
    for r in rows:
        val = r.get(sort_key, "inf")
        try:
            r["_sort_val"] = float(val)
        except (ValueError, TypeError):
            r["_sort_val"] = float("inf")

    rows.sort(key=lambda r: r["_sort_val"])

    # Clean up sort key
    for r in rows:
        r.pop("_sort_val", None)

    # Write sorted CSV
    output_path = Path(args.output or str(input_path.with_suffix("")) + "_sorted.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Sorted leaderboard → {output_path}")
    print(f"  {len(rows)} rows, sorted by {sort_key}")

    # Print top 5
    print("\nTop entries:")
    for i, r in enumerate(rows[:5]):
        exp = r.get("experiment", "?")
        val = r.get(sort_key, "?")
        model = r.get("model_name", "?")
        geo = r.get("geometry_mode", "?")
        status = r.get("status", "?")
        print(f"  {i+1}. {exp} | {model} | {geo} | {sort_key}={val} | {status}")

    # Bar chart
    if args.plot:
        _plot_bar_chart(rows, sort_key, output_path)


def _plot_bar_chart(rows: list[dict], metric: str, output_path: Path) -> None:
    """Generate a horizontal bar chart comparing experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    # Filter to completed only
    completed = [r for r in rows if r.get("status") == "completed"]
    if not completed:
        print("No completed experiments to plot.")
        return

    names = [r.get("experiment", "?") for r in completed]
    vals = [float(r.get(metric, 0)) for r in completed]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    colors = ["#2c7bb6" if v <= min(vals) + 1e-6 else "#abd9e9" for v in vals]
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"Benchmark: {metric}")
    ax.invert_yaxis()

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    plot_path = output_path.with_suffix(".png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart → {plot_path}")


if __name__ == "__main__":
    main()
