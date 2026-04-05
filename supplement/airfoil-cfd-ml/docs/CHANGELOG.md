# Changelog

## [0.2.0] - 2026-04-05

### Added

- Residual learning option in v2 training/evaluation pipeline.
- Optional divergence-based physics regularization.
- Divergence warmup scheduling (`divergence_warmup_ratio`).
- Documentation set for high-quality open-source release:
  - `docs/RESULTS.md`
  - `docs/REPRODUCIBILITY.md`
  - `docs/DATA.md`
  - `docs/CONTRIBUTING.md`
  - `docs/RELEASE_CHECKLIST.md`
  - `docs/PROJECT_STRUCTURE.md`

### Changed

- Expanded experiment tracking with multi-model comparison CSV outputs.
- Promoted `v2_22_twostage_from_v214_lr3e5_e6` to release baseline using fair multi-seed criterion.
- Updated README and docs to align canonical train/eval commands with v2.22 baseline.

## [0.1.0] - Initial

- Baseline U-Net surrogate training and evaluation pipeline.
