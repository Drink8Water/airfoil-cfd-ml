# Reproducibility Guide

## 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 2. Determinism Notes

- Config files include `seed` for data split and training randomness.
- Exact bitwise reproducibility is not guaranteed across CUDA drivers or hardware.

## 3. Canonical Training Runs

### v2.22 release-baseline run

```powershell
python scripts/train_v2.py --config configs/v2_22_twostage_from_v214_lr3e5_e6.yaml
```

### Physics warmup run

```powershell
python scripts/train_v2.py --config configs/v2_3_residual_phys_warmup002.yaml
```

## 4. Canonical Evaluation

```powershell
python scripts/evaluate_v2.py --checkpoint checkpoints_v2_22_twostage_from_v214_lr3e5_e6/dfpnet_best.pt --test_dir ../test
```

## 5. Multi-Seed Fair Comparison (release criterion)

Run each target checkpoint with the same evaluation command and aggregate:

```powershell
python scripts/evaluate_v2.py --checkpoint checkpoints_v2_22_twostage_from_v214_lr3e5_e6/dfpnet_best.pt --test_dir ../test
python scripts/evaluate_v2.py --checkpoint checkpoints_v2_23_twostage_from_v214_lr3e5_e6_seed43/dfpnet_best.pt --test_dir ../test
python scripts/evaluate_v2.py --checkpoint checkpoints_v2_24_twostage_from_v214_lr3e5_e6_seed44/dfpnet_best.pt --test_dir ../test
python scripts/evaluate_v2.py --checkpoint checkpoints_v2_25_twostage_from_v214_lr3e5_e6_seed45/dfpnet_best.pt --test_dir ../test
```

Consolidated output files used in this repository:

- `scan_logs/multiseed_eval_raw.txt`
- `scan_logs/multiseed_eval_latest.csv`

## 6. Unified Multi-Model Comparison

Use the existing comparison scripts or notebook snippets from project history to regenerate CSV leaderboards in checkpoint folders.

## 7. Expected Artifacts per Run

Each `save_dir` should contain:

- `dfpnet_best.pt`
- `epoch_metrics.csv`
- `loss_train.npy`
- `loss_val.npy`
- `training_curve.png`
