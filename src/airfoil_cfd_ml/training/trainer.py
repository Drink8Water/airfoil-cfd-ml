"""Trainer: full training loop with checkpointing, logging, and CSV metrics.

Handles:
  - Deterministic seed
  - Epoch-level train/validate loops
  - Best-checkpoint tracking (val_mean_rel_mae, lower is better)
  - Resolved-config YAML output
  - epoch_metrics.csv
  - train.log text log
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from ..metrics.regression import compute_regression_metrics
from ..utils.device import resolve_device
from .checkpoint import save_checkpoint
from .seed import set_seed


@dataclass
class TrainerConfig:
    """Training hyperparameters and output settings.

    Attributes:
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Adam weight decay.
        seed: Random seed (set at Trainer init).
        save_dir: Checkpoint output directory.
        monitor: Metric name for best-checkpoint selection.
        monitor_mode: ``'min'`` (lower is better) or ``'max'``.
        early_stopping_patience: Epochs without improvement before stopping
            (0 = disabled).
        early_stopping_min_delta: Minimum absolute change to count as
            improvement.
        log_interval: Log every N epochs to train.log (default 1 = every epoch).
        deterministic_cudnn: If True, set cudnn.deterministic=True.
    """

    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    save_dir: str = "outputs/checkpoints"
    monitor: str = "val_mean_rel_mae"
    monitor_mode: str = "min"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 1e-4
    log_interval: int = 1
    deterministic_cudnn: bool = False


def _compute_mean_rel_mae(metrics: Dict[str, float]) -> float:
    """Compute mean_rel_mae = (pressure_rel_mae + u_rel_mae + v_rel_mae) / 3."""
    return (
        metrics.get("pressure_rel_mae", 0.0)
        + metrics.get("u_rel_mae", 0.0)
        + metrics.get("v_rel_mae", 0.0)
    ) / 3.0


class Trainer:
    """Full training loop with logging, checkpointing, and early stopping.

    Usage::

        trainer = Trainer(model, loss_fn, config)
        history = trainer.fit(train_loader, val_loader)
        # best.pt, last.pt, config_resolved.yaml, epoch_metrics.csv,
        # train.log are written to config.save_dir.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainerConfig,
        extra_config: Dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        set_seed(config.seed, deterministic_cudnn=config.deterministic_cudnn)

        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.extra_config = extra_config or {}
        self.device = device or resolve_device()

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.logger = logging.getLogger(f"trainer.{self.save_dir.name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.save_dir / "train.log", mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)
        # Also add a stream handler so output appears in console
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(sh)

        # Resolved config
        self._save_resolved_config()

        # Best tracking
        if config.monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got '{config.monitor_mode}'")
        self._best_value: float = float("inf") if config.monitor_mode == "min" else float("-inf")
        self._best_epoch: int = -1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_resolved_config(self) -> None:
        """Write the resolved training config as YAML."""
        cfg_dict = {
            "epochs": self.config.epochs,
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "seed": self.config.seed,
            "save_dir": str(self.save_dir),
            "monitor": self.config.monitor,
            "monitor_mode": self.config.monitor_mode,
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_min_delta": self.config.early_stopping_min_delta,
        }
        cfg_dict.update(self.extra_config)
        (self.save_dir / "config_resolved.yaml").write_text(
            yaml.dump(cfg_dict, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _is_better(self, value: float) -> bool:
        if self.config.monitor_mode == "min":
            return value < (self._best_value - self.config.early_stopping_min_delta)
        return value > (self._best_value + self.config.early_stopping_min_delta)

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="train", leave=False)
        for batch in pbar:
            x = batch.x.to(self.device)
            y = batch.y.to(self.device)
            mask = batch.mask.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss, _loss_dict = self.loss_fn(pred, y, mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {"train_loss": total_loss / max(len(train_loader), 1)}

    @torch.no_grad()
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        accum: Dict[str, float] = {}

        for batch in val_loader:
            x = batch.x.to(self.device)
            y = batch.y.to(self.device)
            mask = batch.mask.to(self.device)

            pred = self.model(x)
            _, loss_dict = self.loss_fn(pred, y, mask)
            total_loss += loss_dict["total"]

            m = compute_regression_metrics(pred, y, mask)
            for k, v in m.items():
                accum[k] = accum.get(k, 0.0) + v

        n = max(len(val_loader), 1)
        result = {"val_loss": total_loss / n}
        for k, v in accum.items():
            result[k] = v / n
        result["val_mean_rel_mae"] = _compute_mean_rel_mae(result)
        return result

    # ------------------------------------------------------------------
    # Full loop
    # ------------------------------------------------------------------

    def fit(
        self, train_loader, val_loader
    ) -> List[Dict[str, Any]]:
        """Run the full training loop.

        Returns:
            List of per-epoch dicts (same rows written to epoch_metrics.csv).
        """
        csv_path = self.save_dir / "epoch_metrics.csv"
        fieldnames: List[str] = []

        history: List[Dict[str, Any]] = []
        epochs_without_improve = 0

        self.logger.info(f"Trainer start — save_dir={self.save_dir}")
        self.logger.info(f"  epochs={self.config.epochs} lr={self.config.lr} seed={self.config.seed}")

        for epoch in range(self.config.epochs):
            train_info = self.train_epoch(train_loader)
            val_info = self.validate_epoch(val_loader)

            record = {"epoch": epoch + 1}
            record.update(train_info)
            record.update(val_info)
            history.append(record)

            # --- CSV ---
            if not fieldnames:
                fieldnames = list(record.keys())
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(record)

            # --- Log ---
            log_parts = [f"Epoch {epoch+1}/{self.config.epochs}"]
            for k, v in record.items():
                if isinstance(v, float):
                    log_parts.append(f"{k}={v:.6f}")
            self.logger.info(" | ".join(log_parts))

            # --- Best checkpoint ---
            monitor_val = record.get(self.config.monitor)
            if monitor_val is not None and self._is_better(monitor_val):
                self._best_value = monitor_val
                self._best_epoch = epoch + 1
                epochs_without_improve = 0
                save_checkpoint(
                    self.save_dir / "best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    best_value=self._best_value,
                    monitor=self.config.monitor,
                    extra_config=self.extra_config,
                )
                self.logger.info(f"  → new best ({self.config.monitor}={self._best_value:.6f})")
            else:
                epochs_without_improve += 1

            # --- Early stopping ---
            if (
                self.config.early_stopping_patience > 0
                and epochs_without_improve >= self.config.early_stopping_patience
            ):
                self.logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {epochs_without_improve} epochs)"
                )
                break

        # Save last checkpoint
        save_checkpoint(
            self.save_dir / "last.pt",
            model=self.model,
            optimizer=self.optimizer,
            epoch=min(len(history), self.config.epochs),
            extra_config=self.extra_config,
        )
        self.logger.info(f"Training finished. Best {self.config.monitor}={self._best_value:.6f} at epoch {self._best_epoch}")

        return history
