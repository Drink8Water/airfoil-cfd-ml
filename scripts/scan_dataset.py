#!/usr/bin/env python3
"""Dataset scanner: validate all .npz files and emit quality reports.

Produces:
  reports/data_scan.json   — machine-readable per-file validation results.
  docs/DATA_QUALITY.md     — human-readable summary report.

Usage:
  python scripts/scan_dataset.py --data-dir ../train2
  python scripts/scan_dataset.py --data-dir ../test --output-json reports/test_scan.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

# Allow running from repo root without install
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan a directory of .npz airfoil samples and produce quality reports."
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="Path to directory containing .npz files.",
    )
    p.add_argument(
        "--output-json",
        default="reports/data_scan.json",
        help="Path for the JSON report (default: reports/data_scan.json).",
    )
    p.add_argument(
        "--output-md",
        default="docs/DATA_QUALITY.md",
        help="Path for the Markdown report (default: docs/DATA_QUALITY.md).",
    )
    p.add_argument(
        "--expected-spatial",
        nargs=2,
        type=int,
        default=[128, 128],
        help="Expected (H, W). Default: 128 128.",
    )
    return p


def scan_directory(data_dir: str, expected_spatial: tuple[int, int]) -> List[Dict[str, Any]]:
    """Scan all .npz files in a directory and return per-file results."""
    from airfoil_cfd_ml.data.schema import validate_npz_file

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(str(data_path))

    files = sorted(data_path.glob("*.npz"))
    if not files:
        print(f"WARNING: No .npz files found in {data_dir}")
        return []

    results: List[Dict[str, Any]] = []
    n = len(files)

    for i, fp in enumerate(files):
        try:
            result = validate_npz_file(str(fp), expected_spatial=expected_spatial)
        except FileNotFoundError:
            result = {
                "valid": False,
                "path": str(fp),
                "errors": ["File not found (race condition?)"],
                "warnings": [],
                "shape": None,
                "dtype": None,
                "mask_stats": None,
            }
        results.append(result)

        if (i + 1) % max(1, n // 10) == 0:
            print(f"  … {i + 1}/{n} files scanned")

    return results


def build_summary(results: List[Dict[str, Any]], data_dir: str) -> Dict[str, Any]:
    """Aggregate per-file results into a summary dictionary."""
    n_total = len(results)
    if n_total == 0:
        return {"n_total": 0, "n_valid": 0, "n_invalid": 0}

    n_valid = sum(1 for r in results if r["valid"])
    n_invalid = n_total - n_valid

    # Collect shape distribution
    shapes: Dict[str, int] = {}
    dtypes: Dict[str, int] = {}
    for r in results:
        if r["shape"] is not None:
            shapes[str(r["shape"])] = shapes.get(str(r["shape"]), 0) + 1
        if r["dtype"] is not None:
            dtypes[r["dtype"]] = dtypes.get(r["dtype"], 0) + 1

    # Mask binarity
    n_nonbinary = 0
    for r in results:
        ms = r.get("mask_stats")
        if ms and not ms.get("is_binary", True):
            n_nonbinary += 1

    # All errors and warnings
    all_errors: List[str] = []
    all_warnings: List[str] = []
    for r in results:
        for e in r.get("errors", []):
            all_errors.append(f"{Path(r['path']).name}: {e}")
        for w in r.get("warnings", []):
            all_warnings.append(f"{Path(r['path']).name}: {w}")

    return {
        "scan_timestamp_utc": datetime.now(UTC).isoformat(),
        "data_dir": str(Path(data_dir).resolve()),
        "expected_spatial": list(results[0].get("expected_spatial", [None, None]))
        if "expected_spatial" in results[0]
        else None,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "shape_distribution": shapes,
        "dtype_distribution": dtypes,
        "n_nonbinary_mask": n_nonbinary,
        "error_summary": all_errors[:50],  # truncate for JSON readability
        "warning_summary": all_warnings[:50],
    }


def write_markdown_report(summary: Dict[str, Any], md_path: str) -> None:
    """Write a human-readable Markdown report."""
    n = summary.get("n_total", 0)
    n_ok = summary.get("n_valid", 0)
    n_bad = summary.get("n_invalid", 0)

    lines: List[str] = [
        "# Data Quality Report",
        "",
        f"**Generated**: {summary.get('scan_timestamp_utc', 'unknown')}",
        f"**Data directory**: `{summary.get('data_dir', '?')}`",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total files | {n} |",
        f"| Valid | {n_ok} |",
        f"| Invalid | {n_bad} |",
        f"| Non-binary masks | {summary.get('n_nonbinary_mask', '?')} |",
        "",
        "## Shape Distribution",
        "",
    ]

    shapes = summary.get("shape_distribution", {})
    if shapes:
        lines.append("| Shape | Count |")
        lines.append("|---|---|")
        for s, c in sorted(shapes.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {s} | {c} |")
    else:
        lines.append("*(no data)*")

    lines.extend(["", "## Dtype Distribution", ""])
    dtypes = summary.get("dtype_distribution", {})
    if dtypes:
        lines.append("| Dtype | Count |")
        lines.append("|---|---|")
        for d, c in sorted(dtypes.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {d} | {c} |")
    else:
        lines.append("*(no data)*")

    # Errors
    lines.extend(["", "## Errors", ""])
    errors = summary.get("error_summary", [])
    if errors:
        for e in errors:
            lines.append(f"- ❌ {e}")
    else:
        lines.append("✅ No errors.")

    # Warnings
    lines.extend(["", "## Warnings", ""])
    warnings = summary.get("warning_summary", [])
    if warnings:
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append("✅ No warnings.")

    lines.extend(["", "---", "", "*Auto-generated by `scripts/scan_dataset.py`.*", ""])

    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = _make_argparser()
    args = parser.parse_args()

    print(f"Scanning {args.data_dir} …")
    t0 = time.perf_counter()

    expected = tuple(args.expected_spatial) if args.expected_spatial else None
    results = scan_directory(args.data_dir, expected)

    elapsed = time.perf_counter() - t0
    print(f"Scanned {len(results)} files in {elapsed:.1f}s")

    summary = build_summary(results, args.data_dir)

    # Write JSON
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON report → {json_path}")

    # Write Markdown
    write_markdown_report(summary, args.output_md)
    print(f"Markdown report → {args.output_md}")

    # Exit code
    if summary["n_invalid"] > 0:
        print(f"\n⚠️  {summary['n_invalid']}/{summary['n_total']} files are INVALID.")
        sys.exit(1)
    else:
        print(f"\n✅ All {summary['n_total']} files valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()
