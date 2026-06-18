"""CompositeLoss: weighted sum of multiple loss modules.

Each sub-loss must return (total_loss, loss_dict).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """Weighted sum of multiple loss functions.

    Args:
        losses: list of (loss_module, weight) tuples.

    Returns:
        (total_loss, merged_loss_dict)
    """

    def __init__(self, losses: List[Tuple[nn.Module, float]]):
        super().__init__()
        self.sub_losses = nn.ModuleList([l for l, _ in losses])
        self.weights = [w for _, w in losses]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = torch.tensor(0.0, device=pred.device)
        merged: Dict[str, float] = {}
        for loss_fn, w in zip(self.sub_losses, self.weights):
            val, d = loss_fn(pred, target, mask)
            total = total + w * val
            for k, v in d.items():
                merged[k] = v
        merged["total"] = total.item()
        return total, merged
