# AirfoilCFD-ML

**Research-Oriented Scientific Machine Learning for Airfoil Flow Surrogate Modeling**

[![Tests](https://img.shields.io/badge/tests-212%20passed-brightgreen)]()

A reproducible research framework for studying how model inductive bias,
geometry encoding, physics-aware evaluation, and spectral error analysis
affect neural surrogate models for 2D airfoil CFD flow fields.

---

## Project Overview

**Task**: Given airfoil geometry and freestream conditions, predict the
steady-state pressure and velocity fields with a neural network.

| Property | Value |
|----------|-------|
| Input | `[u_inf_x, u_inf_y, mask]` + optional geometry channels |
| Output | `[pressure, u_flow_x, u_flow_y]` |
| Spatial resolution | 128 × 128 |
| Data format | `.npz` with key `a`, shape `(6, 128, 128)` |
| Primary metric | `mean_rel_mae` = (p_rel_mae + u_rel_mae + v_rel_mae) / 3 |

**Core research questions** (see [docs/RESEARCH_QUESTIONS.md](docs/RESEARCH_QUESTIONS.md)):

1. **Model inductive bias** — How do CNN / U-Net / ResU-Net / FNO / GeoFNO-lite / Transolver-lite differ in accuracy-efficiency trade-off?
2. **Geometry encoding** — What is the impact of mask, XY coordinates, SDF, and boundary distance on prediction quality?
3. **Physics-aware evaluation** — Can divergence, vorticity, boundary, and wake metrics reveal weaknesses invisible to pointwise loss?
4. **Spectral error** — Do frequency-domain metrics capture multi-scale structure that per-pixel metrics miss?
5. **Training acceleration** — How much speedup from AMP, torch.compile, channels_last, and LMDB caching?

## Why This Project

- **Not a toy U-Net demo.** A systematic research framework for comparing
  architectures, geometry representations, and evaluation protocols under
  a unified, reproducible pipeline.
- **Config-driven, seed-controlled, artifact-tracked.** Every experiment
  produces `config_resolved.yaml`, `epoch_metrics.csv`, `best.pt`, and
  `eval_metrics.json`.
- **Physics + spectral metrics built in.** Beyond pointwise error, the
  framework computes divergence, vorticity, boundary-region, wake-region,
  and FFT-based spectral errors automatically.
- **Benchmark automation.** `scripts/run_benchmark.py` trains and evaluates
  a directory of configs and produces a ranked CSV leaderboard.

## Current Status

| Module | Status | Notes |
|--------|--------|-------|
| NPZ schema validation | ✅ Implemented | key, shape, dtype, NaN/Inf, mask binarity |
| Geometry encoding | ✅ Implemented | mask, xy, SDF, boundary distance — 4 modes |
| SimpleCNN / ResU-Net | ✅ Implemented | forward/backward tested, smoke-trained |
| FNO2D | ✅ Implemented | unit-tested; **not yet benchmarked** |
| GeoFNO-lite | ✅ Implemented | unit-tested; **not yet benchmarked** |
| Transolver-lite | ✅ Implemented | unit-tested; **not yet benchmarked** |
| Config-driven training | ✅ Implemented | `train.py` + Trainer + early stopping |
| Full evaluation suite | ✅ Implemented | regression + physics + spectral + efficiency |
| Benchmark runner | ✅ Implemented | `run_benchmark.py` + `export_leaderboard.py` |
| v2.22 baseline (legacy) | ✅ Implemented | U-Net + residual + two-stage, 4-seed evaluated |
| Real benchmark results | 📋 Planned | `reports/main_benchmark.csv` does not exist yet |
| Geometry ablation | 📋 Planned | config directory empty |
| Physics loss ablation | 📋 Planned | config directory empty |
| Training acceleration | 📋 Planned | AMP / compile / LMDB not implemented |
| Uncertainty estimation | 📋 Planned | |

## Task Formulation

### Data Format

Each sample is a `.npz` file with key `a` of shape `(6, 128, 128)`:

| Channel index | Role | Description |
|:---:|------|-------------|
| 0 | Input | u_inf_x (freestream x-velocity) |
| 1 | Input | u_inf_y (freestream y-velocity) |
| 2 | Input | mask (0 = fluid, 1 = solid airfoil) |
| 3 | Target | pressure |
| 4 | Target | u_flow_x |
| 5 | Target | u_flow_y |

### Geometry Encoding Modes

The dataset accepts a `geometry_mode` parameter that augments the 3 input
channels with pre-computed geometry features:

| Mode | Channels | Added Features | Status |
|------|:---:|------|:---:|
| `mask_only` | 3 | (none — raw mask only) | ✅ |
| `mask_xy` | 5 | normalized XY coordinates ([-1, 1]) | ✅ |
| `mask_xy_sdf` | 6 | signed distance field (±16 px truncated) | ✅ |
| `mask_xy_sdf_boundary` | 7 | SDF + boundary distance | ✅ |

**SDF convention**: positive in fluid, negative inside solid, zero at the
fluid/solid interface.  Computed via `scipy.ndimage.distance_transform_edt`.
See [docs/GEOMETRY_ENCODING.md](docs/GEOMETRY_ENCODING.md) for details.

## Framework Architecture

```text
src/airfoil_cfd_ml/
├── data/           # AirfoilNPZDataset, schema validation, SDF, geometry
├── models/         # simple_cnn, res_unet, fno2d, geofno_lite, transolver_lite
├── losses/         # FieldLoss (MAE/MSE), CompositeLoss
├── metrics/        # regression, physics (div/vort/boundary/wake),
│                   #   spectral (FFT, energy spectrum), efficiency (params, latency)
├── training/       # Trainer, seed, checkpoint save/load
├── evaluation/     # Evaluator, leaderboard builder
├── utils/          # device resolution
└── visualization/  # (reserved)
```

All models implement `forward(x: (B,C,H,W)) → (B,3,H,W)` and are registered
via `@register_model("name")`.  Build with `build_model(name, **kwargs)`.

## Metrics

The evaluation pipeline computes all of the following automatically.
See [docs/METRICS.md](docs/METRICS.md) for formulas and rationale.

| Category | Metric | Description | Lower is better |
|----------|--------|-------------|:---:|
| Regression | `mean_rel_mae` | mean of channel-wise relative MAE | ✅ |
| Regression | `pressure_rel_mae` | relative MAE, pressure channel | ✅ |
| Regression | `u_rel_mae`, `v_rel_mae` | relative MAE, velocity channels | ✅ |
| Physics | `divergence_error` | mean \|∇·u\| (incompressibility proxy) | ✅ |
| Physics | `vorticity_error` | MAE of vorticity field | ✅ |
| Physics | `boundary_rel_mae` | rel MAE within 3 px of airfoil surface | ✅ |
| Physics | `wake_rel_mae` | rel MAE in wake region | ✅ |
| Spectral | `spectral_error` | FFT amplitude MAE (frequency structure) | ✅ |
| Spectral | `energy_spectrum_error` | radial energy spectrum relative error | ✅ |
| Efficiency | `n_parameters` | trainable parameter count | — |
| Efficiency | `latency_ms_mean` | mean inference latency (ms) | ✅ |

## Quick Start

### 1. Install

```bash
git clone https://github.com/Drink8Water/airfoil-cfd-ml.git
cd airfoil-cfd-ml

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt
```

### 2. Run Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
# 212 passed (expected)
```

### 3. Scan a Dataset Directory

```bash
PYTHONPATH=src python scripts/scan_dataset.py --data-dir ../train2
# Produces: reports/data_scan.json + docs/DATA_QUALITY.md
```

### 4. Smoke Training (1 epoch, minimal network)

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/experiment/smoke_simple_cnn_mask_only.yaml
```

Outputs in `outputs/checkpoints/<experiment_name>/`:

| File | Contents |
|------|----------|
| `best.pt` | best checkpoint (by `val_mean_rel_mae`) |
| `last.pt` | final-epoch checkpoint |
| `config_resolved.yaml` | full resolved experiment config |
| `epoch_metrics.csv` | per-epoch train/val metrics |
| `train.log` | text log |

### 5. Evaluate a Checkpoint

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/smoke_simple_cnn_mask_only/best.pt \
  --config outputs/checkpoints/smoke_simple_cnn_mask_only/config_resolved.yaml
```

Produces `eval_metrics.json` + `per_sample_metrics.csv` in the eval output
directory.

### 6. Benchmark Smoke (pipeline validation only)

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --config-dir configs/experiment/benchmark_smoke \
  --output-csv reports/main_benchmark_smoke.csv
```

> ⚠️ **Smoke results are NOT experimental results.**  They use tiny networks
> (hidden_channels=8), 1 epoch, and synthetic/small data — sufficient only
> for CI/pipeline validation.  Real benchmark configs belong in
> `configs/experiment/main_benchmark/`.

### 7. Export Leaderboard

```bash
PYTHONPATH=src python scripts/export_leaderboard.py \
  --input reports/main_benchmark_smoke.csv --plot
# Writes sorted CSV + optional bar chart PNG
```

## Benchmarking

**No real benchmark results are reported yet.**  The repository currently
provides:

- `configs/experiment/benchmark_smoke/` — 3 configs for smoke-level
  pipeline validation (smoke CSV → `reports/main_benchmark_smoke.csv`).
- `configs/experiment/main_benchmark/` — empty directory for real
  full-training benchmark configs (results would go to
  `reports/main_benchmark.csv`).
- `configs/experiment/geometry_ablation/` — empty directory for geometry
  encoding ablation configs.
- `configs/experiment/physics_ablation/` — empty directory for physics
  loss ablation configs.

Once real benchmark runs are completed, the leaderboard will appear here.
**No numbers are fabricated.**

### v2.22 Legacy Baseline (Reference)

The older v2 pipeline (`train_v2.py`, `evaluate_v2.py`) provides a
reference point with a U-Net + residual learning + two-stage strategy:

| Metric | Value |
|--------|-------|
| Model | DfpNet (U-Net, channel_exponent=6) |
| mean_rel_mae (mean ± std) | **0.573 ± 0.022** |
| Seeds | 42, 43, 44, 45 |

This is the starting baseline for the v3 research framework.  v3 models
(SimpleCNN, ResU-Net, FNO2D, etc.) will be compared against it under the
same evaluation protocol.

## Reproducibility

Every training run produces a complete artifact set:

```text
outputs/checkpoints/<experiment_name>/
├── best.pt                 # best checkpoint by val_mean_rel_mae
├── last.pt                 # final-epoch checkpoint
├── config_resolved.yaml    # full config snapshot
├── epoch_metrics.csv       # per-epoch train/val metrics
├── train.log               # text log
└── eval/
    ├── eval_metrics.json       # aggregate metrics
    └── per_sample_metrics.csv  # per-sample breakdown
```

Key principles:

- Config files include `seed` for data split and training randomness.
- All checkpoints embed their full config and normalization stats.
- Evaluation protocol is identical across all models (same metrics,
  same fluid-region masking).
- See [docs/REPRODUCIBILITY_LINUX.md](docs/REPRODUCIBILITY_LINUX.md)
  for the full reproducibility guide.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/RESEARCH_QUESTIONS.md](docs/RESEARCH_QUESTIONS.md) | Five core research questions |
| [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md) | Model architecture catalog |
| [docs/GEOMETRY_ENCODING.md](docs/GEOMETRY_ENCODING.md) | Geometry encoding schemes |
| [docs/PHYSICS_LOSSES.md](docs/PHYSICS_LOSSES.md) | Physics-aware loss functions |
| [docs/METRICS.md](docs/METRICS.md) | Metric definitions, formulas, and rationale |
| [docs/TRAINING_ACCELERATION.md](docs/TRAINING_ACCELERATION.md) | Training acceleration strategies |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Known limitations and future work |
| [docs/RESULTS.md](docs/RESULTS.md) | Results, leaderboard, and ablations |
| [docs/REPRODUCIBILITY_LINUX.md](docs/REPRODUCIBILITY_LINUX.md) | Linux reproducibility guide |
| [docs/DATA.md](docs/DATA.md) | Data format specification |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Layout and naming conventions |

## Research Roadmap

All items below are **planned** or **in progress** — they are not yet
complete.  Items marked 📋 are accepted as future work.

| Priority | Area | Task | Status |
|----------|------|------|:---:|
| P0 | Benchmark | Run real benchmark for SimpleCNN / ResU-Net / FNO2D | 📋 |
| P0 | Ablation | Geometry encoding ablation (mask → SDF → BD) | 📋 |
| P1 | Ablation | Physics loss ablation | 📋 |
| P1 | Acceleration | AMP mixed-precision training | 📋 |
| P1 | Acceleration | torch.compile integration | 📋 |
| P1 | Acceleration | LMDB dataset cache | 📋 |
| P2 | Analysis | Uncertainty estimation (MC Dropout / Deep Ensemble) | 📋 |
| P2 | Analysis | Failure case analysis by flow condition | 📋 |
| P2 | Benchmark | GeoFNO-lite / Transolver-lite benchmark | 📋 |

## Limitations

- **No real benchmark results yet.**  `reports/main_benchmark.csv` does
  not exist.  Current CSV outputs are smoke-test only.
- **No figures.**  The `figures/` directory is empty.
- **FNO2D / GeoFNO-lite / Transolver-lite** have model-level unit tests
  but **no training benchmark results** — their practical performance on
  this dataset is unknown.
- **Physics metrics are proxies, not CFD validation.**  Low divergence
  error does not guarantee physically correct flow.
- **Smoke tests use synthetic data** (disc obstacles in 64×64 grids).
  Full benchmarks require the real airfoil dataset at 128×128.
- **No SOTA claim.**  This is a research framework, not a claim of
  superiority over any specific method.
- **2D steady flow only.**  3D, unsteady, and compressible effects are
  out of scope.

## Related Work

This project builds on ideas from:

- **FNO** — Li et al., "Fourier Neural Operator for Parametric Partial
  Differential Equations", ICLR 2021.
- **Geo-FNO** — Li et al., "Geometry-Informed Neural Operator for
  Large-Scale 3D PDEs", NeurIPS 2023.
- **Transolver** — Wu et al., "Transolver: A Fast and Accurate
  Transformer-Based PDE Solver", 2024.

The implementations in this repository (`fno2d.py`, `geofno_lite.py`,
`transolver_lite.py`) are lightweight research variants — they are
**not official reproductions** of the original papers.

## License

MIT. See [LICENSE](LICENSE).
