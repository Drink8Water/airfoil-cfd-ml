# scan_logs

This folder stores aggregated outputs from multi-run experiments and evaluation parsing.

## Current files

- `multiseed_eval_raw.txt`: raw concatenated outputs from repeated `scripts/evaluate_v2.py` runs.
- `multiseed_eval_latest.csv`: parsed table used for fair multi-seed ranking.

## Usage notes

- Keep raw logs and parsed CSVs together for traceability.
- Avoid writing model weights or figures in this folder.
- When updating release ranking, regenerate both files in the same run.
