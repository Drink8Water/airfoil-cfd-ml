# Geometry Encoding

本文档描述翼型几何信息的各种编码方式及其对模型性能的影响。

---

## 背景

当前模型通过二值 mask 获取几何信息：

- `mask >= 0.5`：固体/障碍物区域（翼型内部）。
- `mask < 0.5`：流体区域。

Mask 作为输入的第 3 个通道（前 2 个通道为 u_inf_x, u_inf_y）。

**问题**：二值 mask 只提供了"流体/固体"的二分类信息，缺乏：
- 到壁面的距离（边界层效应随距离衰减）。
- 翼型表面法向方向。
- 空间位置参考。

---

## 编码方案总览

| 编码 | 通道数 | 值域 | 计算复杂度 | 预期收益 | 状态 |
|------|--------|------|------------|----------|------|
| Mask (Binary) | 1 | {0, 1} | O(1) | 基线 | ✅ Implemented |
| Mask (Smoothed) | 1 | [0, 1] | O(HW) | 低 | 📋 Planned |
| XY Coordinates | 2 | [-1, 1] | O(1) | 中 | 📋 Planned |
| SDF | 1 | ℝ | O(HW × iter) | 高 | 📋 Planned |
| Boundary Distance | 1 | ℝ⁺ | O(HW) | 高 | 📋 Planned |
| Signed Boundary Distance | 2 | ℝ | O(HW) | 高 | 📋 Planned |
| Multi-scale SDF | 3 | ℝ³ | O(HW × iter × scale) | 最高 | 📋 Planned |

---

## 方案详述

### 1. Binary Mask（已实现）

**当前实现**：

```python
# mask 来自 .npz 数据的第 3 个通道
# 输入: inputs[:, 2:3, :, :] ∈ {0, 1}（通常已经阈值化）
# 流体区域: mask < 0.5, 固体区域: mask >= 0.5
```

**在模型中的使用**：
- 作为前 3 个输入通道之一传入 U-Net。
- 用于计算 spatial weight（fluid_mask = mask < 0.5）。
- 用于障碍物边缘 weight（通过 dilation/erosion 提取 obstacle ring）。

**局限**：
- 模型需要通过卷积层隐式学习几何信息。
- 无法直接表达"距离壁面的远近"。

### 2. Smoothed Mask（计划中）

**定义**：对二值 mask 做高斯模糊，生成 [0, 1] 连续值。

```python
import torch.nn.functional as F

def smoothed_mask(mask_binary: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian smooth the binary mask."""
    kernel_size = int(4 * sigma + 1) | 1  # odd
    # Create Gaussian kernel
    # Apply F.conv2d with Gaussian kernel and reflection padding
    ...
```

**预期收益**：低。主要是消除 mask 边界的锯齿效应。

### 3. XY Coordinates（计划中）

**定义**：将每个像素的归一化空间坐标作为输入通道。

```python
def xy_coordinates(H: int, W: int) -> torch.Tensor:
    """Generate normalized XY coordinate channels.
    
    Returns:
        tensor of shape (1, 2, H, W) with values in [-1, 1].
    """
    yy = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
    xx = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
    return torch.stack([xx, yy], dim=0).unsqueeze(0)
```

**使用**：作为额外 2 个输入通道 concat 到现有输入上，input 从 3 通道变为 5 通道。

**预期收益**：
- 帮助模型打破平移不变性，学习与绝对位置相关的流场特征（如尾流方向）。
- 实现简单，几乎零计算开销。
- 对 FNO 等全局算子可能特别重要（FNO 本身不具备位置感知能力）。

### 4. SDF (Signed Distance Function)（计划中）

**定义**：对每个像素，计算其到最近固体边界的有符号距离。流体区域为正值（在翼型外部），固体区域为负值（在翼型内部）。

```python
import cv2
import numpy as np

def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """Compute signed distance field from binary mask.
    
    Args:
        mask: (H, W) binary array, 1=fluid, 0=solid.
    Returns:
        sdf: (H, W) float array, positive in fluid, negative in solid.
    """
    # Distance to nearest zero (solid boundary)
    dist_fluid = cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    dist_solid = cv2.distanceTransform(
        (1 - mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return dist_fluid - dist_solid
```

**可以预计算并存储在 .npz 中**，避免每次加载时重新计算。

**归一化**：通常将 SDF 截断到 `[-δ, δ]` 范围（如 `δ = 16` 像素）。

**预期收益**：
- 直接提供距离信息，帮助模型建模边界层效应。
- 在 Geo-FNO 等架构中作为几何编码的核心组成部分。
- 预期在壁面附近（±4px 区域）的误差有显著改善。

### 5. Boundary Distance（计划中）

**定义**：到最近壁面的欧氏距离（仅在流体区域有意义）。

```python
def compute_boundary_distance(mask: np.ndarray) -> np.ndarray:
    """Compute distance to nearest wall for each fluid pixel.
    
    Args:
        mask: (H, W) binary array, 1=fluid, 0=solid.
    Returns:
        distance: (H, W) float array.
    """
    return cv2.distanceTransform(
        mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
```

**与 SDF 的区别**：Boundary Distance 仅在流体区域定义，不区分内外。

**预期收益**：与 SDF 类似，但更简单，计算更快。

### 6. Signed Boundary Distance（计划中）

**定义**：结合 SDF 和 Boundary Distance，提供两个独立通道：
- Channel 1: 流体区域到壁面的距离（SDF 的正部）。
- Channel 2: 固体区域到壁面的距离（SDF 的负部的绝对值）。

**预期收益**：比单通道 SDF 信息更丰富，可能对区分"翼型内部"和"翼型外部"有帮助。

### 7. Multi-scale SDF（计划中）

**定义**：计算多个截断尺度下的 SDF，如 `δ ∈ {4, 8, 16}` 像素。

**预期收益**：
- 不同尺度的 SDF 对应不同的物理效应（如 δ=4 捕捉边界层，δ=16 捕捉远场效应）。
- 预期是几何编码的最高收益方案。

---

## 实验设计

### 实验 1：Geometry Encoding Ablation

**目标**：在 DfpNet 上比较不同几何编码的效果。

**固定变量**：
- 模型：DfpNet (channel_exponent=6)
- 训练策略：残差学习
- 损失函数：Weighted L1（无 physics loss）
- Seed：42

**变量**：

| 实验 ID | 几何编码 | 输入通道数 |
|---------|----------|------------|
| geo_baseline | Mask only | 3 |
| geo_xy | Mask + XY | 5 |
| geo_sdf | Mask + SDF (δ=16) | 4 |
| geo_sdf_xy | Mask + SDF (δ=16) + XY | 6 |
| geo_msdf | Mask + Multi-scale SDF | 5 |

**评估**：
- 全局 mean_rel_mae。
- 壁面附近（距边界 ≤4px）的局部 mean_rel_mae。
- 尾流区域的误差分布。

### 实验 2：Geometry × Architecture Interaction

**目标**：不同架构对不同几何编码的敏感度。

在 DfpNet 和 FNO（实现后）上分别比较 mask-only 和 mask+SDF+XY。

---

## 预计算策略

几何编码（SDF, Boundary Distance）可以预计算并存储：

### 方案 A：修改数据预处理

在数据生成阶段计算几何编码，存入 `.npz` 的额外 key（如 `sdf`, `bdist`）。

**优点**：一次计算，永久使用。
**缺点**：需要重新处理所有数据；`.npz` 文件变大。

### 方案 B：在线计算 + 缓存

首次加载时在线计算，结果缓存到内存/磁盘。

```python
class CachedGeometryDataset(Dataset):
    def __init__(self, npz_dir, geo_types=("sdf",)):
        self.npz_dir = Path(npz_dir)
        self.geo_types = geo_types
        self._cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        a = data["a"]
        mask = a[2:3]  # shape (1, H, W)

        geo_channels = []
        if "sdf" in self.geo_types:
            sdf = compute_sdf(mask.squeeze())
            geo_channels.append(torch.from_numpy(sdf).unsqueeze(0))
        ...
        return torch.cat([torch.from_numpy(a[:3]), *geo_channels], dim=0)
```

### 方案 C：LMDB 预计算存储

将所有几何编码预计算后存入 LMDB 数据库（与 RQ4 的 LMDB cache 结合）。

**优点**：最高的 I/O 效率。
**缺点**：实现最复杂。

---

## 实现优先级

1. **XY Coordinates** — 实现最简单，作为几何编码的 quick win。
2. **SDF** — 最重要的几何编码，预期收益最高。
3. **Boundary Distance** — 作为 SDF 的轻量替代方案进行对比。
4. **Multi-scale SDF** — 在前三项验证有效后再实现。

---

*最后更新：2026-06-18*
