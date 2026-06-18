# Results and Ablations

This document tracks reproducible test-set metrics for key checkpoints.

## Evaluation Protocol

- Dataset split and paths are defined by each training config.
- Reported metrics are evaluated on fluid region only.
- Primary ranking metric: mean_rel_mae = (pressure_rel_mae + u_rel_mae + v_rel_mae) / 3.

## Release Decision (Current)

- Release baseline: `v2_22_twostage_from_v214_lr3e5_e6`
- Decision rule: prefer lower multi-seed mean and lower multi-seed std.
- Seed set: `42, 43, 44, 45`

### Fair Multi-Seed Comparison

| Model family | mean(mean_rel_mae) | std(mean_rel_mae) | n | Release status |
|---|---:|---:|---:|---|
| v2.22 (two-stage) | 0.573458 | 0.021663 | 4 | Baseline |
| v2.1 (residual) | 0.646385 | 0.061870 | 4 | Candidate (deprecated) |
| v2.5 (residual + phys-space div) | 0.652976 | 0.083758 | 4 | Rejected |

Source:
- `scan_logs/multiseed_eval_latest.csv`
- `scan_logs/multiseed_eval_raw.txt`

## Historical Single-Run Results (best to worst by mean_rel_mae)

| Model | mean_rel_mae | pressure_rel_mae | u_rel_mae | v_rel_mae | Notes |
|---|---:|---:|---:|---:|---|
| v2_1_residual_w01_full | 0.562336 | 1.007586 | 0.134992 | 0.544430 | Legacy single-run best |
| v2_gradloss_w01_full | 0.621465 | 1.216739 | 0.159113 | 0.488544 | No residual |
| baseline_v1_full | 0.629635 | 1.106426 | 0.170346 | 0.612132 | Original baseline |
| v2_gradloss_full | 0.711365 | 1.362906 | 0.163866 | 0.607324 | Higher grad loss weight |
| v2_gradloss_velonly_full | 0.755288 | 1.461157 | 0.194900 | 0.609807 | Gradient loss excludes pressure |

Source: checkpoints_v2_gradloss/compare_v2_ablation_plus_v21_residual.csv

## Physics-Regularization Trials

| Model | div weight | warmup ratio | mean_rel_mae | Observation |
|---|---:|---:|---:|---|
| v2_2_residual_light_phys_full | 0.02 | 0.0 | 0.751844 | Over-regularized, degraded |
| v2_2_residual_light_phys_w002_full | 0.002 | 0.0 | 0.724363 | Still worse than v2.1 |
| v2_3_residual_phys_warmup002_full | 0.002 | 0.3 | 0.639882 | Better than no-warmup but still below v2.1 |

Sources:
- checkpoints_v2_2_residual_light_phys/compare_with_v21.csv
- checkpoints_v2_2_residual_light_phys_w002/compare_with_v21.csv
- checkpoints_v2_3_residual_phys_warmup002/compare_with_v21.csv

## Key Takeaways

- 发布应以多 seed 公平对比为准，而非单次最优。
- 两阶段版本 v2.22 在均值和方差上均领先，适合作为当前发布基线。
- 残差学习有效，但在稳定性上弱于 v2.22。
