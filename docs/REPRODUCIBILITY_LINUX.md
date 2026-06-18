# Reproducibility Guide — Linux

This document provides the complete Linux command sequence from `git clone` to training,
evaluation, and testing. For the original Windows/PowerShell guide, see
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## 1. System Prerequisites

Verify your hardware and drivers before starting:

```bash
# Check NVIDIA driver and CUDA version
nvidia-smi

# Check CUDA compiler version (if installed)
nvcc --version 2>/dev/null || echo "nvcc not found (PyTorch ships its own CUDA runtime)"

# Check Python version (>=3.10 required)
python3 --version
```

---

## 2. Clone and Environment Setup

### Option A: venv + pip (recommended for most users)

```bash
# Clone the repository
git clone <repo-url> airfoil-cfd-ml
cd airfoil-cfd-ml

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 12.1 example — adjust for your CUDA version)
# See https://pytorch.org/get-started/locally/ for the correct command
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .

# Install dev dependencies (for testing)
pip install -r requirements-dev.txt
```

### Option B: Conda (recommended for GPU training)

```bash
# Clone the repository
git clone <repo-url> airfoil-cfd-ml
cd airfoil-cfd-ml

# Create conda environment from spec file
conda env create -f environment.yml

# Activate the environment
conda activate airfoil-cfd-ml

# Install the project in editable mode
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt
```

#### Conda on a separate partition (when /home is low on space)

If your `/home` partition is small, install the conda environment on a larger partition
by specifying `--prefix`:

```bash
# Example: install to /Extra/conda_envs/airfoil-cfd-ml
conda env create -f environment.yml --prefix /Extra/conda_envs/airfoil-cfd-ml

# Activate by path
conda activate /Extra/conda_envs/airfoil-cfd-ml

# Install the project
pip install -e .
```

Note: PyTorch GPU packages are large (~2–5 GB). Ensure the target partition has at
least 10 GB free before creating the environment.

---

## 3. Verify the Installation

```bash
# 1. Check Python and key packages
python -c "import sys; print('Python:', sys.version)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"

# 2. Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
python -c "import torch; print('Device count:', torch.cuda.device_count())"
python -c "import torch; print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 3. Verify the project package is importable
python -c "from airfoil_cfd_ml.model import DfpNet; print('v1 DfpNet OK')"
python -c "from airfoil_cfd_ml_v2.model import DfpNet; print('v2 DfpNet OK')"

# 4. Run smoke tests (no GPU required)
pytest tests/ -v

# 5. Quick forward-pass test
python -c "
import torch
from airfoil_cfd_ml.model import DfpNet
model = DfpNet(channel_exponent=4)
x = torch.randn(2, 3, 128, 128)
y = model(x)
print(f'Input: {x.shape} -> Output: {y.shape}')
print('Forward pass OK')
"
```

Expected output for a working CUDA setup:

```
CUDA available: True
CUDA version: 12.1
Device count: 1
Device name: NVIDIA GeForce RTX 3060
```

---

## 4. Prepare Data

The config files expect data directories relative to the project root:

```bash
# Expected layout (data is NOT in the repository):
#   ../train2/   — training .npz files
#   ../test/     — test .npz files

# If your data is elsewhere, either:
#   (a) Symlink it:
#       ln -s /path/to/your/train2 ../train2
#       ln -s /path/to/your/test ../test
#
#   (b) Or edit the train_dir / test_dir paths in config YAML files.
```

---

## 5. Training

### v2.22 release-baseline (two-stage from v2.14 checkpoint)

```bash
python scripts/train_v2.py --config configs/v2_22_twostage_from_v214_lr3e5_e6.yaml
```

### Physics warmup run

```bash
python scripts/train_v2.py --config configs/v2_3_residual_phys_warmup002.yaml
```

### Quick smoke test (shorter training, for CI/validation)

```bash
python scripts/train_v2.py --config configs/smoke_edge_earlystop.yaml
```

### CPU-only training

```bash
python scripts/train_v2.py --config configs/v2_22_twostage_from_v214_lr3e5_e6.yaml --cpu
```

---

## 6. Evaluation

### Single checkpoint evaluation

```bash
python scripts/evaluate_v2.py \
  --checkpoint checkpoints_v2_22_twostage_from_v214_lr3e5_e6/dfpnet_best.pt \
  --test_dir ../test
```

### Multi-seed fair comparison (release criterion)

```bash
# Evaluate all four seeds of the v2.22 family
for seed in v2_22 v2_23_seed43 v2_24_seed44 v2_25_seed45; do
    dir=$(ls -d checkpoints_${seed}* 2>/dev/null | head -1)
    if [ -n "$dir" ]; then
        echo "=== Evaluating $dir ==="
        python scripts/evaluate_v2.py \
          --checkpoint "$dir/dfpnet_best.pt" \
          --test_dir ../test
    fi
done
```

---

## 7. Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=airfoil_cfd_ml --cov=airfoil_cfd_ml_v2 --cov-report=term-missing
```

---

## 8. Determinism Notes

- Config files include `seed` for data split and training randomness.
- PyTorch determinism can be enforced with:
  ```python
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
- Exact bitwise reproducibility is **not guaranteed** across different CUDA drivers,
  GPU architectures, or PyTorch versions.

---

## 9. Expected Artifacts per Training Run

Each `save_dir` should contain:

| File | Description |
|------|-------------|
| `dfpnet_best.pt` | Best model checkpoint (state dict + stats + config) |
| `epoch_metrics.csv` | Per-epoch training/validation metrics |
| `loss_train.npy` | Training loss history (NumPy array) |
| `loss_val.npy` | Validation loss history (NumPy array) |
| `training_curve.png` | Training/validation loss plot |

---

## 10. Troubleshooting

| Symptom | Check |
|---------|-------|
| `torch.cuda.is_available() == False` | `nvidia-smi` works? Driver version ≥ PyTorch CUDA version? |
| `No .npz files found` | Data directories exist and contain `.npz` files? |
| `ImportError: No module named 'airfoil_cfd_ml'` | Did you run `pip install -e .`? |
| `ModuleNotFoundError: No module named 'torch'` | PyTorch installed? Check `pip list \| grep torch` |
| CUDA out of memory | Reduce `batch_size` in the YAML config |

---

## 11. Quick-Start One-Liner (after clone + venv)

```bash
pip install --upgrade pip && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
pip install -r requirements.txt && \
pip install -e . && \
pip install -r requirements-dev.txt && \
pytest tests/ -v && \
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('All checks passed')"
```
