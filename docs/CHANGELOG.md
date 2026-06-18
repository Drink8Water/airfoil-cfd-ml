# Changelog

## [0.3.0] - 2026-06-18

### Added

- Rewritten README in Chinese with research-oriented framing, implemented/in-progress/planned status tags.
- `docs/RESEARCH_QUESTIONS.md` — five core research questions with experimental designs.
- `docs/MODEL_ZOO.md` — model architecture catalog with interface specs and planned architectures.
- `docs/GEOMETRY_ENCODING.md` — geometry encoding schemes (mask, XY, SDF, boundary distance) with implementation plans.
- `docs/PHYSICS_LOSSES.md` — physics-aware loss functions documentation (implemented + planned).
- `docs/TRAINING_ACCELERATION.md` — training acceleration strategies (AMP, torch.compile, channels_last, LMDB, DataLoader).
- `docs/LIMITATIONS.md` — honest assessment of known limitations, failure modes, and future roadmap.

### Changed

- README: repositioned from course-project to research-oriented SciML framework.
- README: clearly distinguishes implemented / in-progress / planned features.
- PROJECT_STRUCTURE.md: updated to reflect new docs.

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
