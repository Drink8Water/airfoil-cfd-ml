# Project Structure

This document defines the repository layout and naming conventions used for experiments and release artifacts.

## Top-Level Layout

```text
airfoil-cfd-ml/
  configs/        # YAML experiment definitions
  docs/           # project documentation
  notebooks/      # exploratory analysis notebooks
  scan_logs/      # aggregated experiment/evaluation logs
  scripts/        # CLI entrypoints (train/evaluate)
  src/            # Python packages (v1 and v2 pipelines)
  tests/          # smoke and regression checks
  checkpoints*/   # experiment outputs (weights + curves + per-epoch metrics)
```

## Checkpoint Folder Convention

Checkpoint directories follow this pattern:

- `checkpoints_<experiment_name>/`

Each directory should contain:

- `dfpnet_best.pt`
- `epoch_metrics.csv`
- `loss_train.npy`
- `loss_val.npy`
- `training_curve.png`

Optional files:

- `figures_compare/` for qualitative GT/Pred/Error figures
- `compare_*.csv` for side-by-side model comparisons

## Config Naming Convention

- Base variants: `v2_<id>_<short_description>.yaml`
- Seed variants: append `_seedXX` (for example: `..._seed43.yaml`)
- Two-stage variants: include `twostage` in name

This naming keeps sorting stable and links each config to a checkpoint directory.

## Release Baseline Artifacts (Current)

- Baseline family: `v2_22_twostage_from_v214_lr3e5_e6`
- Main checkpoint: `checkpoints_v2_22_twostage_from_v214_lr3e5_e6/dfpnet_best.pt`
- Seed replicas:
  - `checkpoints_v2_23_twostage_from_v214_lr3e5_e6_seed43/`
  - `checkpoints_v2_24_twostage_from_v214_lr3e5_e6_seed44/`
  - `checkpoints_v2_25_twostage_from_v214_lr3e5_e6_seed45/`
- Aggregated fair-comparison logs:
  - `scan_logs/multiseed_eval_raw.txt`
  - `scan_logs/multiseed_eval_latest.csv`

## Practical Guidance

- Keep generated artifacts inside dedicated `checkpoints_*` or `scan_logs` paths.
- Do not mix code and experiment output in `src/`.
- When adding a new release candidate, update:
  - `README.md`
  - `docs/RESULTS.md`
  - `docs/REPRODUCIBILITY.md`
  - `docs/CHANGELOG.md`