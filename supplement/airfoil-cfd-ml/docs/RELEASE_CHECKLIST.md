# Release Checklist

Use this checklist before making the repository public.

## Legal and Metadata

- [x] License file exists and matches intended usage.
- [ ] No private/proprietary data is committed.
- [x] README clearly states dataset origin and redistribution limits.

## Documentation

- [x] README has quickstart, training, evaluation, and project scope.
- [x] README release baseline section points to current checkpoint.
- [x] `docs/RESULTS.md` is updated with latest leaderboard.
- [x] `docs/REPRODUCIBILITY.md` commands are verified.
- [x] `docs/DATA.md` reflects actual format and paths.
- [x] `docs/PROJECT_STRUCTURE.md` matches current folder conventions.

## Code Health

- [x] `python -m compileall src` passes.
- [x] Main train/eval scripts run without path issues.
- [x] Config files for key experiments are present and named clearly.

## Experiments

- [x] Release checkpoint is identified in docs.
- [x] Release decision includes multi-seed mean/std table.
- [x] At least one failed ablation is documented with conclusion.
- [x] Qualitative figure examples are included.

## Repository Hygiene

- [ ] Large artifacts are intentionally included or excluded.
- [x] `.gitignore` is reviewed.
- [ ] Commit history is understandable.
- [x] Aggregated release logs exist in `scan_logs/` (or are reproducible by commands in docs).

## Verification Notes (2026-04-05)

- Automated checks completed: `python -m compileall src`, baseline `scripts/evaluate_v2.py` command, path/link existence scan for README + docs.
- Remaining manual checks: private/proprietary data audit, large artifact policy decision, final commit-history review.
