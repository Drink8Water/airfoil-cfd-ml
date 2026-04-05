# Airfoil CFD ML

面向工程落地的 CFD surrogate 项目：基于 U-Net 风格神经网络进行 2D 翼型流场代理预测（pressure / u / v），并以多 seed 公平评估确定可发布基线。

## Why This Project

- 目标：用神经网络近似 CFD 结果，降低推理成本并保持可解释的误差评估流程。
- 方法：围绕残差学习、物理约束、两阶段训练进行系统化实验。
- 结果：在统一评估协议下，v2.22 取得当前最优的均值与稳定性。

## My Contributions

- 重构训练与评估流水线（`src/airfoil_cfd_ml_v2`），形成配置驱动实验框架。
- 建立多 seed 公平对比流程，使用均值和方差而非单次最优做发布决策。
- 实现并对比残差学习、散度约束、两阶段训练等策略。
- 补齐项目文档体系（结果、复现、结构、发布检查清单）。

## Project Background

本仓库基于南京大学南赫学院本科专业课《地球流体力学》课程项目重构，后续按工程化标准完善为可复现、可比较、可扩展的实践仓库。

## Release Baseline

- Baseline model: `v2_22_twostage_from_v214_lr3e5_e6`
- Checkpoint: `checkpoints_v2_22_twostage_from_v214_lr3e5_e6/dfpnet_best.pt`
- Fair multi-seed result (seed 42/43/44/45):
  - mean_rel_mae mean = `0.573458`
  - mean_rel_mae std = `0.021663`
- Mean metric definition: `(pressure_rel_mae + u_rel_mae + v_rel_mae) / 3`

说明：单次最优值不是发布依据，当前以多 seed 的均值和稳定性作为发布标准。

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

训练（当前发布基线同系列配置）：

```powershell
python scripts/train_v2.py --config configs/v2_22_twostage_from_v214_lr3e5_e6.yaml
```

评估：

```powershell
python scripts/evaluate_v2.py --checkpoint checkpoints_v2_22_twostage_from_v214_lr3e5_e6/dfpnet_best.pt --test_dir ../test
```

测试可视化对比图（v2.22）：

- `checkpoints_v2_22_twostage_from_v214_lr3e5_e6/figures_compare/compare_sample000_pressure.png`
- `checkpoints_v2_22_twostage_from_v214_lr3e5_e6/figures_compare/compare_sample010_pressure.png`
- `checkpoints_v2_22_twostage_from_v214_lr3e5_e6/figures_compare/compare_sample020_pressure.png`

详细结果和多 seed 排名见 `docs/RESULTS.md`。

## Project Layout

```text
airfoil-cfd-ml/
  configs/
  docs/
    PROJECT_STRUCTURE.md
  scripts/
  src/
    airfoil_cfd_ml/
    airfoil_cfd_ml_v2/
  scan_logs/
  tests/
  README.md
  pyproject.toml
  requirements.txt
```

## Data Format

- Default paths: `../train2`, `../test`
- Dataset origin: course-provided CFD samples used for academic project experiments.
- Redistribution note: raw dataset files are not bundled in this repository; ensure you have redistribution rights before publishing any data copies.
- One sample per `.npz` with key `a`, shape `(6, 128, 128)`
- Channels:
  - Input: `[u_inf_x, u_inf_y, mask]`
  - Target: `[pressure, u_flow_x, u_flow_y]`

详见 `docs/DATA.md`。

## Documentation Index

- Results and ablations: `docs/RESULTS.md`
- Reproducibility commands: `docs/REPRODUCIBILITY.md`
- Repository layout and naming conventions: `docs/PROJECT_STRUCTURE.md`
- Data specification: `docs/DATA.md`
- Contribution guide: `docs/CONTRIBUTING.md`
- Release checklist: `docs/RELEASE_CHECKLIST.md`
- Changelog: `docs/CHANGELOG.md`

## Known Findings

- 两阶段策略（v2.22）在公平四 seed 下取得最优均值与最佳稳定性。
- 残差学习明显优于早期基线。
- 固定权重散度约束在当前设置下提升有限，且稳定性不如两阶段基线。

## License

MIT. See `LICENSE`.
