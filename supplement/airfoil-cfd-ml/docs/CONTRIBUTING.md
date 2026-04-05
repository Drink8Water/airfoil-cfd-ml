# Contributing

Thanks for your interest in improving this project.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Typical Workflow

1. Create a branch for your change.
2. Keep changes focused (one topic per PR).
3. Run smoke checks before opening a PR.
4. Update documentation for behavior/config changes.

## Suggested Checks

```powershell
python -m compileall src
python scripts/evaluate_v2.py --checkpoint checkpoints_v2_22_twostage_from_v214_lr3e5_e6/dfpnet_best.pt --test_dir ../test
```

## Repository Structure Rules

- Put new configs in `configs/` with stable versioned names.
- Keep all training outputs in `checkpoints_<experiment_name>/`.
- Put aggregated metrics/log parsing outputs in `scan_logs/`.
- Update `docs/PROJECT_STRUCTURE.md` if naming conventions change.

## Pull Request Expectations

- Clear summary of what changed and why.
- Before/after metrics for model-related changes.
- Mention config and checkpoint used for experiments.
- Add notes to `docs/RESULTS.md` when introducing new ablations.
- If release baseline changes, update `README.md` and `docs/REPRODUCIBILITY.md` in the same PR.
