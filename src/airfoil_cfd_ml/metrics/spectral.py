"""Spectral-domain evaluation metrics using 2D real FFT.

Compares the amplitude spectra of predicted and target fields.
"""

from __future__ import annotations

from typing import Optional

import torch


def _ensure_mask_4d(
    mask: Optional[torch.Tensor], like: torch.Tensor
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask


def spectral_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Mean absolute error between the amplitude spectra of pred and target.

    Uses ``torch.fft.rfft2`` and compares the magnitudes in frequency space.

    Args:
        pred: (B, 3, H, W).
        target: (B, 3, H, W).
        mask: Optional (B, 1, H, W) fluid mask.  If given, solid regions
            are zeroed before the FFT (reduces ringing).

    Returns:
        Scalar float: average MAE of amplitude spectra across channels.
    """
    mask = _ensure_mask_4d(mask, pred)

    if mask is not None:
        pred = pred * mask
        target = target * mask

    # Per-channel FFT
    errors = []
    for c in range(3):
        amp_pred = torch.fft.rfft2(pred[:, c:c+1]).abs()
        amp_tgt = torch.fft.rfft2(target[:, c:c+1]).abs()
        ch_err = (amp_pred - amp_tgt).abs().mean().item()
        errors.append(ch_err)

    return float(sum(errors) / 3.0)


def energy_spectrum_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Mean relative error of the radially-averaged energy spectrum.

    The energy spectrum E(k) is the total power at wavenumber magnitude k,
    averaged over all orientations.  We compare the normalised spectra of
    pred and target per channel, then take the mean across channels.

    Args:
        pred: (B, 3, H, W).
        target: (B, 3, H, W).
        mask: Optional fluid mask.

    Returns:
        Scalar float: mean relative error of the energy spectrum.
    """
    mask = _ensure_mask_4d(mask, pred)

    if mask is not None:
        pred = pred * mask
        target = target * mask

    B, C, H, W = pred.shape

    # Pre-compute wavenumber magnitude bins for rfft2 output shape
    rfft_h, rfft_w = H, W // 2 + 1
    ky = torch.fft.fftfreq(H, d=1.0).abs()[:rfft_h]
    kx = torch.fft.rfftfreq(W, d=1.0)[:rfft_w]
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    k_mag = torch.sqrt(KX**2 + KY**2)  # (H, W//2+1)
    k_floor = k_mag.floor().long()
    # Number of unique integer k values
    num_bins = int(k_floor.max().item()) + 1

    total_rel_err = 0.0
    n_valid = 0

    for b in range(B):
        for c in range(C):
            amp_pred = torch.fft.rfft2(pred[b, c:c+1]).abs().squeeze()  # (H, W//2+1)
            amp_tgt = torch.fft.rfft2(target[b, c:c+1]).abs().squeeze()

            # Radially bin
            e_pred = torch.zeros(num_bins, device=pred.device)
            e_tgt = torch.zeros(num_bins, device=pred.device)
            counts = torch.zeros(num_bins, device=pred.device)

            for ki in range(num_bins):
                sel = k_floor == ki
                c_sel = sel.sum()
                if c_sel > 0:
                    e_pred[ki] = amp_pred[sel].mean()
                    e_tgt[ki] = amp_tgt[sel].mean()
                    counts[ki] = c_sel

            # Relative error per bin, averaged
            diff = (e_pred - e_tgt).abs()
            denom = e_tgt.abs().clamp_min(1e-8)
            rel_err = (diff / denom).mean().item()
            total_rel_err += rel_err
            n_valid += 1

    if n_valid == 0:
        return 0.0

    return float(total_rel_err / n_valid)
