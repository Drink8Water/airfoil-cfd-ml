"""Leaderboard: collect eval results and build a ranked comparison table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def collect_eval_results(
    output_root: str,
) -> List[Dict[str, Any]]:
    """Collect all ``eval_metrics.json`` files under ``output_root``.

    Args:
        output_root: Root directory containing experiment sub-directories.
            Each sub-directory that contains an ``eval/eval_metrics.json``
            file will be included.

    Returns:
        List of dicts with keys: experiment, config_file, metrics...
    """
    rows: List[Dict[str, Any]] = []
    root = Path(output_root)

    # Walk all eval_metrics.json files
    for json_path in sorted(root.rglob("eval_metrics.json")):
        exp_dir = json_path.parent.parent  # .../experiment/eval/ → experiment/
        exp_name = exp_dir.name

        try:
            metrics = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        row: Dict[str, Any] = {"experiment": exp_name, "eval_dir": str(json_path.parent)}
        # Try to get config info from parent dir
        resolved_yaml = exp_dir / "config_resolved.yaml"
        if resolved_yaml.exists():
            try:
                import yaml
                cfg = yaml.safe_load(resolved_yaml.read_text(encoding="utf-8"))
                row["model_name"] = cfg.get("model_name", "?")
                row["geometry_mode"] = cfg.get("geometry_mode", "?")
            except Exception:
                row["model_name"] = "?"
                row["geometry_mode"] = "?"

        row.update(metrics)
        rows.append(row)

    return rows


def build_leaderboard(
    checkpoint_dirs: List[str],
    sort_by: str = "mean_rel_mae",
) -> List[Dict[str, Any]]:
    """Build a leaderboard by collecting eval results from multiple directories.

    Args:
        checkpoint_dirs: List of directory paths to scan for eval_metrics.json.
        sort_by: Metric key to sort by (ascending).

    Returns:
        List of rows sorted by ``sort_by``.
    """
    all_rows: List[Dict[str, Any]] = []
    for d in checkpoint_dirs:
        all_rows.extend(collect_eval_results(d))

    # Sort
    all_rows.sort(key=lambda r: float(r.get(sort_by, float("inf"))))
    return all_rows


def save_leaderboard_csv(
    rows: List[Dict[str, Any]],
    output_path: str,
    columns: Optional[List[str]] = None,
) -> str:
    """Save leaderboard rows to CSV.

    Args:
        rows: List of metric dicts.
        output_path: Path to write CSV.
        columns: Columns to include (default: auto-detect from rows).

    Returns:
        The output path.
    """
    if not rows:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("", encoding="utf-8")
        return output_path

    if columns is None:
        # Heuristic: metadata columns first, then metrics
        meta_cols = ["experiment", "model_name", "geometry_mode", "eval_dir"]
        metric_cols = [k for k in rows[0].keys() if k not in meta_cols]
        columns = [c for c in meta_cols if c in rows[0]] + sorted(metric_cols)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return output_path
