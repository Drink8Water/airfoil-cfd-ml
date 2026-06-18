# Model Zoo

本文档记录项目中所有模型的架构细节、参数规模、输入输出规范和实现状态。

---

## 统一接口规范

所有模型必须满足以下接口约定：

```python
# Input:  x ∈ R^{B × C × H × W}
# Output: y ∈ R^{B × 3 × H × W}
# 前 3 个输出通道分别对应: pressure, u_flow_x, u_flow_y
```

模型通过 register 装饰器注册到全局 registry：

```python
from airfoil_cfd_ml.models.registry import register_model, build_model

@register_model("my_model")
class MyModel(BaseModel):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # returns: (B, 3, H, W)
        ...

# 构建模型
model = build_model("my_model", **kwargs)
```

---

## 已实现模型

### DfpNet (U-Net)

| 属性 | 值 |
|------|-----|
| 类型 | CNN Encoder-Decoder with skip connections |
| 状态 | ✅ Implemented |
| 文件 | `src/airfoil_cfd_ml_v2/model.py` |
| 输入 | `(B, 3, 128, 128)` — [u_inf_x, u_inf_y, mask] |
| 输出 | `(B, 3, 128, 128)` — [pressure, u_flow_x, u_flow_y] |
| 参数量 | ~7.8M (channel_exponent=6, base_channels=64) |
| 关键超参 | channel_exponent, dropout |
| 当前 baseline | v2.22, mean_rel_mae ≈ 0.573 (4 seed mean) |

**架构描述**：

- 7 层 encoder（stride-2 conv + BatchNorm + LeakyReLU），通道数从 64 翻倍至 512。
- 7 层 decoder（upsample + conv + BatchNorm + ReLU），skip connection 拼接 encoder 输出。
- 最终层：1×1 conv 投影到 3 个输出通道，无 BatchNorm 无激活。

**使用示例**：

```python
from airfoil_cfd_ml_v2.model import DfpNet
model = DfpNet(channel_exponent=6, dropout=0.0)
```

### SimpleCNN

| 属性 | 值 |
|------|-----|
| 类型 | 浅层 CNN baseline |
| 状态 | ✅ Implemented |
| 文件 | `src/airfoil_cfd_ml/models/simple_cnn.py` |
| 输入 | `(B, C, H, W)` — 可配置 |
| 输出 | `(B, 3, H, W)` |
| 参数量 | 可配置（默认 hidden_channels=32, n_layers=4 → ~35K） |
| 关键超参 | in_channels, hidden_channels, n_layers, out_channels |

**架构描述**：

- N 层 conv (3×3, same padding) + BatchNorm + ReLU。
- 最终 3×3 conv 投影到 3 个输出通道。
- 空间分辨率保持不变。

**使用示例**：

```python
from airfoil_cfd_ml.models.registry import build_model
model = build_model("simple_cnn", in_channels=3, hidden_channels=32, n_layers=4)
```

**用途**：

- 框架 smoke test 的快速验证模型。
- 新损失函数 / 训练策略的快速原型验证。
- 作为最小 viable baseline 对比更复杂的架构。

---

## 计划中的模型

### ResU-Net

| 属性 | 值 |
|------|-----|
| 类型 | U-Net with residual blocks |
| 状态 | 📋 Planned |
| 预期输入 | `(B, 3, 128, 128)` |
| 预期输出 | `(B, 3, 128, 128)` |
| 预期参数量 | ~8–10M |

**设计思路**：

- 将 DfpNet 的 encoder/decoder block 替换为残差 block（两个 3×3 conv + skip connection）。
- 保持 U-Net 的 skip connection 架构。
- 预期在深层网络中改善梯度流动和收敛速度。

**关键设计选择**（待定）：
- Bottleneck vs basic residual block？
- Pre-activation vs post-activation？
- 是否在 skip connection 中也加 1×1 conv 对齐通道数？

### FNO (Fourier Neural Operator)

| 属性 | 值 |
|------|-----|
| 类型 | Neural Operator (Fourier space) |
| 状态 | 📋 Planned |
| 预期输入 | `(B, C, 128, 128)` |
| 预期输出 | `(B, 3, 128, 128)` |
| 预期参数量 | ~2–5M (取决于 modes 和 width) |

**设计思路**：

- 输入通过 lifting layer 提升到高维空间。
- 多个 Fourier layer：FFT → 截断高频 modes → 线性变换 → IFFT + 局部 conv。
- 最终 projection layer 映射到 3 个输出通道。

**关键设计选择**（待定）：
- Fourier modes 数量：对 128×128 网格，初步计划保留 12–16 个 modes。
- 是否使用 FNO-2D 或 FNO-3D（将通道维度视为第三维）？
- 如何处理 mask（固体区域）— padding 还是 masking？

**参考文献**：
- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.

### Geo-FNO-lite

| 属性 | 值 |
|------|-----|
| 类型 | Geometry-Aware FNO |
| 状态 | 📋 Planned |
| 预期输入 | `(B, C+geo, 128, 128)` — 额外几何编码通道 |
| 预期输出 | `(B, 3, 128, 128)` |
| 预期参数量 | ~2–6M |

**设计思路**：

- 在 FNO 基础上，将几何编码（SDF / boundary distance / xy coords）作为额外输入通道。
- 或者：在 Fourier layer 之前通过一个几何编码网络将坐标映射到高维。
- "Lite" 版本：简化 FNO 结构，减少 Fourier modes 和层数，使模型更轻量。

**关键设计选择**（待定）：
- 几何编码是 concat 到输入还是在内部注入？
- 是否需要特殊的 padding 处理不规则几何？

**参考文献**：
- Li et al., "Geometry-Informed Neural Operator for Large-Scale 3D PDEs", NeurIPS 2023.

### Transolver-lite

| 属性 | 值 |
|------|-----|
| 类型 | Transformer-based Physics Solver |
| 状态 | 📋 Planned |
| 预期输入 | `(B, C, 128, 128)` |
| 预期输出 | `(B, 3, 128, 128)` |
| 预期参数量 | ~5–15M |

**设计思路**：

- 将 2D 场视为 patch sequence（类似 ViT）。
- 加入物理信息（坐标编码、边界条件）作为 positional encoding。
- 使用 self-attention 建模全局依赖。
- "Lite" 版本：减少层数/头数/embedding 维度，适合 128×128 的中等分辨率。

**关键设计选择**（待定）：
- Patch size：8×8（256 patches）vs 16×16（64 patches）？
- 是否需要上采样层恢复全分辨率？
- 如何在 attention 中融入物理先验？

**参考文献**：
- Wu et al., "Transolver: A Fast and Accurate Transformer-Based PDE Solver", 2024.

---

## 模型对比矩阵

| 模型 | 参数量 (预估) | 推理时间 (预估) | 归纳偏置 | 长程依赖 | 几何感知 |
|------|---------------|-----------------|----------|----------|----------|
| SimpleCNN | ~35K | 快 | 局部 | ❌ | ❌ |
| DfpNet (U-Net) | ~7.8M | 中 | 多尺度局部 | 有限(skip) | 隐式(mask) |
| ResU-Net | ~8–10M | 中 | 多尺度局部+残差 | 有限(skip) | 隐式(mask) |
| FNO | ~2–5M | 中-快 | 全局(Fourier) | ✅ | ❌ |
| Geo-FNO-lite | ~2–6M | 中-快 | 全局+几何 | ✅ | ✅ |
| Transolver-lite | ~5–15M | 慢 | 全局(attention) | ✅ | 可设计 |

> **注意**：推理时间和参数量为预估，实际数值将在实现后更新。不要引用此表作为 benchmark 结果。

---

## 添加新模型

1. 在 `src/airfoil_cfd_ml/models/` 下创建新文件。
2. 继承 `BaseModel`，实现 `forward(x) → (B, 3, H, W)`。
3. 使用 `@register_model("model_name")` 注册。
4. 在 `tests/test_models.py` 中添加 smoke test（input/output shape 验证）。
5. 更新本文档（状态从 📋 Planned 变为 ✅ Implemented + 实际参数量 + 实际指标）。

---

*最后更新：2026-06-18*
