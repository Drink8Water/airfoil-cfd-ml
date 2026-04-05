# Airfoil CFD ML

课程项目重构版：使用 U-Net 风格神经网络做 2D 翼型流场代理预测（pressure / u / v）。

本仓库重点是“可复现 + 可比较 + 可扩展”，包含基线、残差学习、轻量物理约束等实验分支与完整记录。

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
