"""NpzAeroDataset (legacy) and AirfoilNPZDataset (enhanced) for .npz airfoil data.

AirfoilNPZDataset supports geometry_mode for multi-channel geometry encoding:
  mask_only, mask_xy, mask_xy_sdf, mask_xy_sdf_boundary.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import GEOMETRY_MODES, augment_geometry_channels, get_input_channels
from .schema import DataSample, validate_npz_file


class NpzAeroDataset(Dataset):
    """Legacy dataset for .npz airfoil CFD data (mask_only mode, 3-channel input).

    Maintained for backward compatibility.  Prefer AirfoilNPZDataset for new work.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_list = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        )
        if not self.file_list:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> DataSample:
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        arr = data["a"]  # (6, 128, 128)

        x = torch.from_numpy(arr[:3, :, :].astype(np.float32))
        y = torch.from_numpy(arr[3:, :, :].astype(np.float32))
        mask = (x[2:3, :, :] < 0.5).float()

        return DataSample(
            x=x,
            y=y,
            mask=mask,
            meta={"file_name": self.file_list[idx], "index": idx},
        )


class AirfoilNPZDataset(Dataset):
    """Enhanced dataset for .npz airfoil CFD data with geometry encoding.

    Each .npz file contains key ``'a'`` with shape ``(6, 128, 128)``:
      Channels 0–2: input  ``[u_inf_x, u_inf_y, mask]``
      Channels 3–5: target ``[pressure, u_flow_x, u_flow_y]``

    Geometry modes (see docs/GEOMETRY_ENCODING.md):
      ========================  ===  ===========================================
      mode                      Cin  channels
      ========================  ===  ===========================================
      mask_only                 3    u_inf_x, u_inf_y, mask
      mask_xy                   5    + x_coord, y_coord (normalised [-1,1])
      mask_xy_sdf               6    + truncated SDF (normalised [-1,1])
      mask_xy_sdf_boundary      7    + truncated boundary distance (norm. [0,1])
      ========================  ===  ===========================================

    Args:
        data_dir: Path to directory containing .npz files.
        geometry_mode: One of the modes above (default ``"mask_only"``).
        sdf_truncate: Truncation distance in pixels for SDF / boundary distance.
        validate: If True, validate each .npz on construction (warns on issues).
    """

    def __init__(
        self,
        data_dir: str,
        geometry_mode: str = "mask_only",
        sdf_truncate: float = 16.0,
        validate: bool = False,
    ):
        if geometry_mode not in GEOMETRY_MODES:
            raise ValueError(
                f"Unknown geometry_mode '{geometry_mode}'. "
                f"Choose from {sorted(GEOMETRY_MODES)}."
            )

        self.data_dir = os.path.abspath(data_dir)
        self.geometry_mode = geometry_mode
        self.sdf_truncate = float(sdf_truncate)
        self.input_channels = get_input_channels(geometry_mode)

        self.file_list = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        )
        if not self.file_list:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        # Optional validation pass
        self._validation_errors: list[str] = []
        if validate:
            self._run_validation()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> DataSample:
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        arr = data["a"]  # (6, 128, 128)

        x_raw = torch.from_numpy(arr[:3, :, :].astype(np.float32))  # (3, H, W)
        y = torch.from_numpy(arr[3:, :, :].astype(np.float32))  # (3, H, W)

        # Augment with geometry channels
        x = augment_geometry_channels(
            x_raw, mode=self.geometry_mode, sdf_truncate=self.sdf_truncate
        )

        # Fluid mask: 1=fluid, 0=solid
        mask = (x_raw[2:3, :, :] < 0.5).float()

        return DataSample(
            x=x,
            y=y,
            mask=mask,
            meta={
                "file_name": self.file_list[idx],
                "index": idx,
                "geometry_mode": self.geometry_mode,
                "input_channels": self.input_channels,
            },
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _run_validation(self) -> None:
        """Validate every .npz and collect errors/warnings."""
        for fname in self.file_list:
            fpath = os.path.join(self.data_dir, fname)
            result = validate_npz_file(fpath)
            if not result["valid"]:
                self._validation_errors.append(
                    f"{fname}: {'; '.join(result['errors'])}"
                )
            for w in result["warnings"]:
                self._validation_errors.append(f"{fname}: [WARN] {w}")

    @property
    def is_valid(self) -> bool:
        """True if no validation errors were found."""
        return len(self._validation_errors) == 0

    @property
    def validation_report(self) -> list[str]:
        """List of validation error/warning strings."""
        return list(self._validation_errors)
