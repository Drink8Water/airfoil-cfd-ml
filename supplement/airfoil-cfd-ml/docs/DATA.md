# Data Specification

## Directory Layout

By default, this project expects paths relative to repository root:

- `../train2`
- `../test`

These paths are configurable through YAML config files.

## File Format

Each sample is an `.npz` file with key `a` and shape `(6, 128, 128)`.

- Input channels: `[u_inf_x, u_inf_y, mask]`
- Target channels: `[pressure, u_flow_x, u_flow_y]`

## Mask Convention

- `mask < 0.5`: fluid region
- `mask >= 0.5`: solid/obstacle region

Training loss and reported metrics are computed over fluid region (plus spatial weighting rules from config).

## Data Governance

- Do not commit private/raw datasets to this repository unless redistribution rights are confirmed.
- If sharing publicly, provide either:
  - a download script, or
  - instructions to recreate from an open source dataset.
