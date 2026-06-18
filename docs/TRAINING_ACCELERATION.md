# Training Acceleration

本文档描述训练加速策略的设计、实现和 benchmark 计划。

---

## 动机

当前 DfpNet (U-Net) 在 RTX 3060（12GB）上的训练时间约为：
- 单 epoch：约 30–60 秒（取决于 batch size 和硬件）。
- 完整训练（50 epochs + early stopping）：约 20–40 分钟。
- Multi-seed (4 seeds)：约 1.5–3 小时。

随着模型复杂度增加（FNO, Transformer），训练时间可能增长数倍。系统性地优化训练速度可以加速研究迭代周期。

---

## 加速策略总览

| 策略 | 预期加速 | 精度影响 | 实现难度 | 状态 |
|------|----------|----------|----------|------|
| DataLoader 调优 | 1.5–3× (I/O bound) | 无 | 低 | 🚧 部分可配置 |
| AMP (FP16/BF16) | 1.5–2.5× | 极小（需监控） | 低 | 📋 Planned |
| torch.compile | 1.2–1.8× | 无 | 低 | 📋 Planned |
| channels_last | 1.1–1.4× | 无 | 中（需改代码） | 📋 Planned |
| LMDB Cache | 3–10× (I/O bound) | 无 | 中 | 📋 Planned |
| Gradient Accumulation | 等效增大 batch | 极小 | 低 | 📋 Planned |

---

## 策略详述

### 1. DataLoader 调优

**状态**：🚧 部分可配置（`num_workers`, `pin_memory` 已暴露），但未系统 benchmark。

**可调参数**：

```yaml
# config YAML 中对应字段
batch_size: 20       # 当前默认值
num_workers: 0       # 当前默认值（0 = 主进程加载）
```

**优化建议**：

| 参数 | 当前值 | 推荐值 | 理由 |
|------|--------|--------|------|
| num_workers | 0 | 4–8 | 多进程并行加载 .npz 文件 |
| prefetch_factor | (默认 2) | 4 | 每个 worker 预取更多 batch |
| pin_memory | True (CUDA) | True | 加速 CPU→GPU 传输 |
| persistent_workers | False | True | 避免每个 epoch 重新 fork workers |

**实验计划**：

```python
# Benchmark script pseudocode
for num_workers in [0, 2, 4, 8]:
    for batch_size in [16, 20, 32, 64]:
        measure_throughput(num_workers, batch_size)
```

**预期**：
- 当前 num_workers=0 意味着数据加载和训练在同一个进程中串行执行。
- 设置 num_workers=4 预期显著减少 GPU 空闲等待时间。
- 受限于 .npz 文件的磁盘 I/O（机械硬盘 vs SSD 差异很大）。

### 2. AMP (Automatic Mixed Precision)

**状态**：📋 Planned

**实现方案**：

```python
# 使用 torch.cuda.amp
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    optimizer.zero_grad()
    with autocast():  # 自动选择 FP16/BF16
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**注意事项**：
- FP16 的数值范围有限，某些操作（如 softmax, loss computation）需要在 FP32 下执行。
- 建议使用 BF16（如果 GPU 支持，如 A100/RTX 4090），数值范围与 FP32 相同。
- RTX 3060 不支持 BF16，只能使用 FP16。
- 需要确保 divergence loss 等自定义操作在 FP16 下数值稳定。

**预期加速**：
- RTX 3060 (FP16)：1.5–2.0×
- A100/4090 (BF16)：2.0–2.5×

### 3. torch.compile

**状态**：📋 Planned

**实现方案**：

```python
# PyTorch 2.0+
model = torch.compile(model, mode="reduce-overhead")
# 或
model = torch.compile(model, mode="max-autotune")  # 首次编译慢，后续快
```

**注意事项**：
- 首次调用时会触发 JIT 编译，第一个 epoch 可能较慢。
- `mode="max-autotune"` 首次编译非常慢但后续最快。
- 可能与自定义操作（如 `_build_border_taper` 中的动态 shape）冲突。
- 需要 PyTorch >= 2.0。

**预期加速**：1.2–1.8×（取决于模型结构和 batch size）。

### 4. channels_last (NHWC)

**状态**：📋 Planned

**动机**：NVIDIA GPU 的 Tensor Cores 对 NHWC 内存布局有更好的支持。

**实现方案**：

```python
# 将模型和输入转换为 channels_last
model = model.to(memory_format=torch.channels_last)
inputs = inputs.to(memory_format=torch.channels_last)
```

**注意事项**：
- 需要在整个数据流中保持一致的内存格式。
- `F.interpolate`（U-Net 中的 upsample）在 channels_last 下的行为需要验证。
- BatchNorm 和 Conv2d 在 channels_last 下通常更快。

**预期加速**：1.1–1.4×。

### 5. LMDB Cache

**状态**：📋 Planned

**动机**：
- 每个 .npz 文件只有约 100KB（(6, 128, 128) × 4 bytes ≈ 393KB 压缩后），但文件数量多时，文件系统的 open/read/close 开销可能成为瓶颈。
- LMDB (Lightning Memory-Mapped Database) 将数据预加载到单个数据库文件中，支持零拷贝读取。

**实现方案**：

```python
import lmdb
import numpy as np

class LmdbAeroDataset(Dataset):
    def __init__(self, lmdb_path: str):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = np.frombuffer(txn.get(f"{idx:06d}".encode()), dtype=np.float32)
            a = data.reshape(6, 128, 128)
        ...

# 构建 LMDB 的脚本
def build_lmdb(npz_dir: str, lmdb_path: str):
    env = lmdb.open(lmdb_path, map_size=int(1e10))  # 10GB
    with env.begin(write=True) as txn:
        for i, npz_file in enumerate(sorted(Path(npz_dir).glob("*.npz"))):
            data = np.load(npz_file)["a"].astype(np.float32)
            txn.put(f"{i:06d}".encode(), data.tobytes())
```

**注意事项**：
- 需要在训练前运行一次构建脚本。
- LMDB 文件大小 ≈ 数据总大小（6 × 128 × 128 × 4 bytes × N_samples）。
- 可以同时存储几何编码（SDF 等），一劳永逸。

**预期加速**：
- 机械硬盘 (HDD)：5–10× I/O 加速。
- SSD：2–5× I/O 加速。
- NVMe：1.5–3× I/O 加速。

### 6. Gradient Accumulation

**状态**：📋 Planned

**动机**：当 GPU 内存不足以支持理想 batch size 时，通过梯度累积模拟更大的 batch size。

**实现方案**：

```python
accumulation_steps = 4  # 有效 batch_size = config_batch_size × 4
for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**注意事项**：
- BatchNorm 的统计量在小 batch 下不准确。考虑使用 GroupNorm 或 SyncBatchNorm。
- 对 divergence loss 等依赖空间导数的损失，小 batch 的影响需要评估。

---

## Benchmark 计划

### Benchmark 脚本

```bash
python scripts/benchmark_training.py \
  --config configs/benchmark/baseline.yaml \
  --strategies amp compile channels_last lmdb dataloader \
  --epochs 3 \
  --output reports/benchmark_training_$(date +%Y%m%d).csv
```

### 记录指标

| 指标 | 说明 |
|------|------|
| Total wall time | 完整训练的端到端时间 |
| Per-epoch time | 每 epoch 平均时间 |
| Data loading time | 数据加载耗时（profiler） |
| GPU utilization | `nvidia-smi` 采集的 GPU 利用率 |
| GPU memory peak | 峰值 GPU 内存使用 |
| Final val loss | 确保加速不损害精度 |
| Throughput | samples/second |

---

## 兼容性矩阵

| 策略 | CUDA 11.x | CUDA 12.x | PyTorch 1.x | PyTorch 2.x | RTX 3060 | A100 |
|------|-----------|-----------|-------------|-------------|----------|------|
| DataLoader | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AMP (FP16) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AMP (BF16) | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ |
| torch.compile | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| channels_last | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| LMDB Cache | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 实现优先级

1. **DataLoader 调优** — 最低风险，可能带来最大收益（当前 num_workers=0 是已知瓶颈）。
2. **AMP (FP16)** — 实现简单，加速效果显著。
3. **LMDB Cache** — 需要额外构建步骤，但对 I/O 密集型训练改善大。
4. **torch.compile** — PyTorch 2.0 特性，一键启用。
5. **channels_last** — 收益较小但实现简单，可与 AMP 叠加。
6. **Gradient Accumulation** — 仅在需要增大有效 batch size 时使用。

---

*最后更新：2026-06-18*
