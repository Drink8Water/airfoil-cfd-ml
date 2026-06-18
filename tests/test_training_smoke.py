"""Smoke tests for training pipeline: loss, metrics, trainer end-to-end.

Includes:
  - Fake-dataset tests (no .npz files needed) — fast.
  - Synthetic-npz tests (tempfile .npz files) — full pipeline coverage.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from airfoil_cfd_ml.data.schema import DataSample, collate_stack
from airfoil_cfd_ml.data.dataset_npz import AirfoilNPZDataset
from airfoil_cfd_ml.models.simple_cnn import SimpleCNN
from airfoil_cfd_ml.models.res_unet import ResUNet
from airfoil_cfd_ml.losses.field import FieldLoss
from airfoil_cfd_ml.losses.composite import CompositeLoss
from airfoil_cfd_ml.metrics.regression import compute_regression_metrics
from airfoil_cfd_ml.training.trainer import Trainer, TrainerConfig
from airfoil_cfd_ml.training.seed import set_seed
from airfoil_cfd_ml.evaluation.evaluator import Evaluator


# ======================================================================
# Helpers
# ======================================================================

CANONICAL_KEY = "a"


def _make_synthetic_npz_dir(
    base_dir: str,
    n_files: int = 8,
    h: int = 64,
    w: int = 64,
) -> str:
    """Create a directory of synthetic .npz files with disc obstacles.

    Returns the path to the directory.
    """
    data_dir = os.path.join(base_dir, "synthetic_data")
    os.makedirs(data_dir, exist_ok=True)

    for i in range(n_files):
        arr = np.zeros((6, h, w), dtype=np.float32)
        arr[0] = np.float32(0.8)  # u_inf_x
        arr[1] = np.float32(0.0)  # u_inf_y
        # disc obstacle at centre
        yy, xx = np.ogrid[:h, :w]
        cx, cy, r = w // 2, h // 2, 8
        solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        arr[2] = solid.astype(np.float32)
        arr[3] = np.random.randn(h, w).astype(np.float32) * 0.1
        arr[4] = np.random.randn(h, w).astype(np.float32) * 5.0
        arr[5] = np.random.randn(h, w).astype(np.float32) * 5.0
        np.savez_compressed(
            os.path.join(data_dir, f"sample_{i:04d}.npz"),
            **{CANONICAL_KEY: arr},
        )
    return data_dir


# ---------------------------------------------------------------------------
# Fake dataset (no I/O)
# ---------------------------------------------------------------------------
class _FakeAeroDataset:
    def __len__(self):
        return 8

    def __getitem__(self, idx: int):
        return DataSample(
            x=torch.randn(3, 64, 64),
            y=torch.randn(3, 64, 64),
            mask=torch.ones(1, 64, 64),
            meta={"idx": idx},
        )


# ======================================================================
# FieldLoss
# ======================================================================

def test_field_loss_returns_tuple():
    loss_fn = FieldLoss(reduction="mse")
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    total, d = loss_fn(pred, target, mask)
    assert isinstance(total, torch.Tensor)
    assert total.ndim == 0
    assert "total" in d
    assert "loss_pressure" in d
    assert "loss_u" in d
    assert "loss_v" in d


def test_field_loss_mae_reduction():
    loss_fn = FieldLoss(reduction="mae")
    pred = torch.zeros(2, 3, 64, 64)
    target = torch.ones(2, 3, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    _total, d = loss_fn(pred, target, mask)
    assert 0.5 < d["total"] < 2.0


def test_field_loss_masked_ignores_solid():
    loss_fn = FieldLoss(reduction="mse")
    pred = torch.randn(1, 3, 8, 8)
    target = pred.clone()
    mask = torch.zeros(1, 1, 8, 8)
    _total, d = loss_fn(pred, target, mask)
    assert d["total"] == pytest.approx(0.0, abs=1e-6)


# ======================================================================
# CompositeLoss
# ======================================================================

def test_composite_loss():
    l1 = FieldLoss(reduction="mse")
    l2 = FieldLoss(reduction="mae")
    comp = CompositeLoss([(l1, 1.0), (l2, 0.5)])
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    mask = torch.ones(2, 1, 32, 32)
    total, d = comp(pred, target, mask)
    assert isinstance(total, torch.Tensor)
    assert "total" in d


# ======================================================================
# Metrics
# ======================================================================

def test_metrics_returns_expected_keys():
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    m = compute_regression_metrics(pred, target, mask)
    expected = {
        "pressure_mae", "pressure_rmse", "pressure_rel_mae",
        "u_mae", "u_rmse", "u_rel_mae",
        "v_mae", "v_rmse", "v_rel_mae",
    }
    assert set(m.keys()) == expected
    for v in m.values():
        assert isinstance(v, float)
        assert v >= 0.0


# ======================================================================
# Trainer smoke (fake data, 1 epoch)
# ======================================================================

def test_trainer_one_epoch():
    ds = _FakeAeroDataset()
    train_loader = DataLoader(ds, batch_size=4, collate_fn=collate_stack)
    val_loader = DataLoader(ds, batch_size=4, collate_fn=collate_stack)

    model = SimpleCNN(in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
    loss_fn = FieldLoss(reduction="mse")
    config = TrainerConfig(epochs=1, lr=0.01, save_dir="outputs/test_trainer")

    trainer = Trainer(model, loss_fn, config)
    history = trainer.fit(train_loader, val_loader)

    assert len(history) == 1
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
    assert isinstance(history[0]["train_loss"], float)


def test_trainer_saves_checkpoint():
    ds = _FakeAeroDataset()
    train_loader = DataLoader(ds, batch_size=4, collate_fn=collate_stack)
    val_loader = DataLoader(ds, batch_size=4, collate_fn=collate_stack)

    model = SimpleCNN(in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
    loss_fn = FieldLoss(reduction="mse")
    save_dir = "outputs/test_trainer_ckpt"
    config = TrainerConfig(epochs=1, lr=0.01, save_dir=save_dir)

    trainer = Trainer(model, loss_fn, config)
    trainer.fit(train_loader, val_loader)

    assert os.path.isfile(os.path.join(save_dir, "last.pt"))


# ======================================================================
# Trainer — full outputs (best.pt, config_resolved.yaml, csv, log)
# ======================================================================

def test_trainer_outputs_all_artifacts():
    """Trainer.fit produces best.pt, last.pt, config_resolved.yaml,
    epoch_metrics.csv, train.log."""
    ds = _FakeAeroDataset()
    train_loader = DataLoader(ds, batch_size=4, collate_fn=collate_stack)
    val_loader = DataLoader(ds, batch_size=4, collate_fn=collate_stack)

    model = SimpleCNN(in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
    loss_fn = FieldLoss(reduction="mse")
    save_dir = "outputs/test_full_outputs"
    config = TrainerConfig(epochs=2, lr=0.01, seed=42, save_dir=save_dir)

    trainer = Trainer(model, loss_fn, config)
    trainer.fit(train_loader, val_loader)

    # Check all expected outputs
    assert os.path.isfile(os.path.join(save_dir, "best.pt")), "best.pt missing"
    assert os.path.isfile(os.path.join(save_dir, "last.pt")), "last.pt missing"
    assert os.path.isfile(os.path.join(save_dir, "config_resolved.yaml")), "config_resolved.yaml missing"
    assert os.path.isfile(os.path.join(save_dir, "epoch_metrics.csv")), "epoch_metrics.csv missing"
    assert os.path.isfile(os.path.join(save_dir, "train.log")), "train.log missing"

    # Verify CSV has 2 rows + header
    with open(os.path.join(save_dir, "epoch_metrics.csv"), "r") as f:
        lines = f.readlines()
    assert len(lines) == 3, f"Expected header + 2 data rows, got {len(lines)}"

    # Verify config YAML is parseable
    import yaml
    with open(os.path.join(save_dir, "config_resolved.yaml"), "r") as f:
        resolved = yaml.safe_load(f)
    assert resolved["seed"] == 42
    assert resolved["epochs"] == 2


# ======================================================================
# Trainer — best.pt selection (val_mean_rel_mae)
# ======================================================================

def test_trainer_saves_best_by_mean_rel_mae():
    """Trainer monitors val_mean_rel_mae (lower is better)."""
    ds = _FakeAeroDataset()
    train_loader = DataLoader(ds, batch_size=4, collate_fn=collate_stack)
    val_loader = DataLoader(ds, batch_size=4, collate_fn=collate_stack)

    model = SimpleCNN(in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
    loss_fn = FieldLoss(reduction="mse")
    save_dir = "outputs/test_best_selection"
    config = TrainerConfig(epochs=3, lr=0.01, seed=123, save_dir=save_dir)

    trainer = Trainer(model, loss_fn, config)
    trainer.fit(train_loader, val_loader)

    assert os.path.isfile(os.path.join(save_dir, "best.pt"))


# ======================================================================
# Synthetic NPZ smoke tests — full pipeline
# ======================================================================

class TestPipelineWithSyntheticNPZ:
    """Full pipeline: AirfoilNPZDataset → SimpleCNN → Trainer.fit → evaluate."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = _make_synthetic_npz_dir(self._tmpdir.name, n_files=8, h=64, w=64)
        self.val_dir = _make_synthetic_npz_dir(
            os.path.join(self._tmpdir.name, "val"), n_files=4, h=64, w=64
        )
        yield
        self._tmpdir.cleanup()

    def test_simple_cnn_mask_only_1_epoch(self):
        """SimpleCNN + mask_only: 1 epoch, check outputs."""
        ds = AirfoilNPZDataset(self.data_dir, geometry_mode="mask_only")
        val_ds = AirfoilNPZDataset(self.val_dir, geometry_mode="mask_only")
        train_loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_stack)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_stack)

        model = SimpleCNN(in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
        loss_fn = FieldLoss(reduction="mse")
        save_dir = os.path.join(self._tmpdir.name, "outputs", "smoke_npz")
        config = TrainerConfig(epochs=1, lr=0.01, seed=42, save_dir=save_dir)

        trainer = Trainer(model, loss_fn, config)
        history = trainer.fit(train_loader, val_loader)

        assert len(history) == 1
        assert os.path.isfile(os.path.join(save_dir, "best.pt"))
        assert os.path.isfile(os.path.join(save_dir, "last.pt"))
        assert os.path.isfile(os.path.join(save_dir, "config_resolved.yaml"))
        assert os.path.isfile(os.path.join(save_dir, "epoch_metrics.csv"))
        assert os.path.isfile(os.path.join(save_dir, "train.log"))

        # Verify data sample structure
        sample = ds[0]
        assert sample.x.shape == (3, 64, 64)
        assert sample.y.shape == (3, 64, 64)
        assert sample.mask.shape == (1, 64, 64)

    def test_simple_cnn_mask_xy_sdf_1_epoch(self):
        """SimpleCNN + mask_xy_sdf (6 channels): 1 epoch, check outputs."""
        ds = AirfoilNPZDataset(self.data_dir, geometry_mode="mask_xy_sdf")
        val_ds = AirfoilNPZDataset(self.val_dir, geometry_mode="mask_xy_sdf")
        train_loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_stack)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_stack)

        model = SimpleCNN(in_channels=6, hidden_channels=8, n_layers=2, out_channels=3)
        loss_fn = FieldLoss(reduction="mse")
        save_dir = os.path.join(self._tmpdir.name, "outputs", "smoke_npz_sdf")
        config = TrainerConfig(epochs=1, lr=0.01, seed=42, save_dir=save_dir)

        trainer = Trainer(model, loss_fn, config)
        history = trainer.fit(train_loader, val_loader)

        assert len(history) == 1
        # Verify geometry channels: input should be 6-channel
        sample = ds[0]
        assert sample.x.shape == (6, 64, 64)

    def test_res_unet_mask_only_1_epoch(self):
        """ResUNet + mask_only: 1 epoch, check outputs."""
        ds = AirfoilNPZDataset(self.data_dir, geometry_mode="mask_only")
        val_ds = AirfoilNPZDataset(self.val_dir, geometry_mode="mask_only")
        train_loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_stack)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_stack)

        model = ResUNet(in_channels=3, channel_exponent=4)
        loss_fn = FieldLoss(reduction="mse")
        save_dir = os.path.join(self._tmpdir.name, "outputs", "smoke_resunet")
        config = TrainerConfig(epochs=1, lr=0.01, seed=42, save_dir=save_dir)

        trainer = Trainer(model, loss_fn, config)
        history = trainer.fit(train_loader, val_loader)

        assert len(history) == 1
        assert os.path.isfile(os.path.join(save_dir, "best.pt"))

    def test_evaluate_after_training(self):
        """Train → evaluate → check eval_metrics.json exists."""
        ds = AirfoilNPZDataset(self.data_dir, geometry_mode="mask_only")
        val_ds = AirfoilNPZDataset(self.val_dir, geometry_mode="mask_only")
        train_loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_stack)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_stack)

        model = SimpleCNN(in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
        loss_fn = FieldLoss(reduction="mse")
        save_dir = os.path.join(self._tmpdir.name, "outputs", "eval_test")
        config = TrainerConfig(epochs=1, lr=0.01, seed=42, save_dir=save_dir)

        trainer = Trainer(model, loss_fn, config)
        trainer.fit(train_loader, val_loader)

        # Now evaluate on val set
        eval_output_dir = os.path.join(self._tmpdir.name, "outputs", "eval_results")
        evaluator = Evaluator(model)
        metrics = evaluator.evaluate(val_loader, output_dir=eval_output_dir)

        assert os.path.isfile(os.path.join(eval_output_dir, "eval_metrics.json"))
        assert os.path.isfile(os.path.join(eval_output_dir, "per_sample_metrics.csv"))
        assert "mean_rel_mae" in metrics
        assert metrics["mean_rel_mae"] >= 0.0

        # Verify JSON content
        with open(os.path.join(eval_output_dir, "eval_metrics.json"), "r") as f:
            json_metrics = json.load(f)
        assert "mean_rel_mae" in json_metrics

        # Verify CSV content
        with open(os.path.join(eval_output_dir, "per_sample_metrics.csv"), "r") as f:
            lines = f.readlines()
        assert len(lines) == len(val_ds) + 1  # header + 4 samples


# ======================================================================
# Seed reproducibility
# ======================================================================

def test_set_seed_deterministic():
    """Same seed + same init → same outputs."""
    set_seed(42)
    model1 = SimpleCNN(in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
    set_seed(42)
    model2 = SimpleCNN(in_channels=3, hidden_channels=8, n_layers=2, out_channels=3)
    x = torch.randn(1, 3, 16, 16)
    set_seed(99)  # reset RNG for inference
    with torch.no_grad():
        y1 = model1(x)
        y2 = model2(x)
    assert torch.allclose(y1, y2), "Same seed should produce identical models"
