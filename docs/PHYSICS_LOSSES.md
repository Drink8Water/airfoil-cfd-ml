# Physics-Aware Losses

本文档描述项目中所有物理约束损失函数的设计、实现和实验结果。

---

## 设计原则

1. **模块化**：每个物理损失是独立的 `nn.Module`，可自由组合。
2. **可配置权重**：所有权重通过 YAML config 控制。
3. **Mask 感知**：所有损失仅在流体区域计算（通过 spatial_weight 实现）。
4. **归一化感知**：支持在归一化空间或物理空间中计算（`divergence_use_physical_space` flag）。

---

## 已实现的物理损失

### 1. WeightedMultiChannelLoss（通道加权 L1 Loss）

| 属性 | 值 |
|------|-----|
| 文件 | `src/airfoil_cfd_ml_v2/losses.py` |
| 状态 | ✅ Implemented |

**公式**：

```
L_value = mean( w_c * |pred_c - target_c| * spatial_weight )
```

其中 `w = [5.0, 1.0, 1.0]` 对 pressure/u/v 分别加权。

**设计理由**：
- Pressure 的数值范围远小于速度，需要更高的权重来平衡学习。
- 权重 5:1:1 是经验选择，尚未做系统的权重扫描。

**实现要点**：
- 支持可选的 `mask` 和 `spatial_weight`。
- `mask` 限制在流体区域计算。
- `spatial_weight` 在 mask 基础上进一步加权（边界 taper + 障碍物边缘）。

### 2. GradientDifferenceLoss（梯度差异损失）

| 属性 | 值 |
|-----|-----|
| 文件 | `src/airfoil_cfd_ml_v2/losses.py` |
| 状态 | ✅ Implemented |

**公式**：

```
L_grad = 0.5 * ( MAE(∂_x pred, ∂_x target) + MAE(∂_y pred, ∂_y target) )
```

使用一阶有限差分计算空间梯度。

**物理意义**：
- 鼓励预测场与目标场在空间变化率上一致。
- 主要对速度场的 smoothness 有帮助。

**当前发现**：
- weight 为 0.1–0.2 时有效提升速度场预测。
- weight 过大会伤害 pressure 预测。
- 在 v2.22 中 weight 为 0.1。

### 3. Divergence Loss（散度损失）

| 属性 | 值 |
|-----|-----|
| 文件 | `src/airfoil_cfd_ml_v2/losses.py` |
| 状态 | ✅ Implemented |

**公式**（归一化空间）：

```
L_div = mean( |∂_x u_norm + scale * ∂_y v_norm| * spatial_weight )
```

**物理意义**：
- 不可压缩流动应满足 ∇·u = ∂u/∂x + ∂v/∂y = 0。
- 此损失惩罚模型输出中违反连续性方程的部分。

**高级特性**：

1. **物理空间计算** (`divergence_use_physical_space: true`)：
   - 先将归一化速度 denormalize 到物理单位，再计算散度。
   - 物理空间的散度值具有更明确的物理意义。

2. **dv/dy 缩放** (`divergence_dvdy_scale`)：
   - 对 ∂v/∂y 项单独缩放，用于处理各向异性网格或流动特征。
   - v2.22 中使用默认值 1.0。

3. **Interior Margin** (`divergence_interior_margin`)：
   - 仅在流体区域内部（距 mask 边界至少 margin 像素）计算散度。
   - v2.22 中使用 margin=6，避免边界附近数值误差干扰。

4. **Divergence Warmup** (`divergence_warmup_ratio`)：
   - 训练前 warmup_ratio 比例的 epochs 不施加 divergence loss。
   - 让模型先学会基本的流场预测，再加入物理约束。
   - v2.22 中 warmup_ratio=0.7。

5. **Adaptive Divergence** (`adaptive_divergence: true/false`)：
   - 基于 validation pressure_rel_mae 自适应调整 divergence 权重。
   - 如果 pressure 误差大 → 降低 divergence weight（优先拟合数据）。
   - 如果 pressure 误差小 → 提高 divergence weight（加强物理约束）。
   - 实验显示 adaptive 在稳定性上优于固定权重。

### 4. CompositeV2Loss（组合损失）

| 属性 | 值 |
|-----|-----|
| 文件 | `src/airfoil_cfd_ml_v2/losses.py` |
| 状态 | ✅ Implemented |

**公式**：

```
L_total = L_value + w_grad * L_gradient + w_div * L_divergence
```

所有子损失的权重通过 YAML config 控制。

---

## 计划中的物理损失

### 5. Boundary Loss（边界条件损失）

| 状态 | 📋 Planned |

**动机**：在翼型表面，真实流动满足无滑移条件（速度为零）。当前模型无法显式保证这一点。

**公式**（草案）：

```
L_boundary = mean_{pixels in boundary_ring} |pred_velocity|
```

其中 `boundary_ring` 是 mask 边界周围的一个窄带（1–2 像素宽）。

**实现要点**：
- 通过 morphological dilation/erosion 提取边界 ring。
- 只需对速度通道（输出通道 2 和 3）施加。
- 可能需要与 divergence loss 协调权重，避免冲突。

### 6. Wake Loss（尾流加权损失）

| 状态 | 📋 Planned |

**动机**：翼型尾流区域是预测最困难也最重要的区域。标准 L1 loss 对所有流体区域等权重处理，尾流区域的误差可能被稀释。

**公式**（草案）：

```
L_wake = mean( wake_weight * |pred - target| ) / mean(wake_weight)
```

其中 `wake_weight` 在尾流方向（来流方向下游）增大。

**实现要点**：
- 需要自动确定尾流区域（基于来流方向和翼型位置）。
- 可以对不同下游距离使用指数衰减权重。
- 需要与 primary loss 协调：是替换还是额外补充？

### 7. Spectral Loss（频谱损失）

| 状态 | 📋 Planned |

**动机**：湍流流动在不同空间频率上有不同的能量分布（能量级联）。频谱损失可以帮助模型更好地捕捉湍流的频谱特征。

**公式**（草案）：

```
L_spectral = MAE( |FFT(pred)|, |FFT(target)| )
```

或使用 2D FFT 的径向平均功率谱。

**实现要点**：
- 仅对速度场（或 pressure+速度场）计算。
- mask 处理：在固体区域填 0 可能导致 FFT 出现 ringing 效应。可能需要 smooth filling。
- 可以选择在 log-spectrum 或 linear spectrum 上计算。

**参考文献**：
- Chattopadhyay et al., "Data-driven super-parameterization using deep learning", 2020.

### 8. Multi-order Gradient Loss（多阶梯度损失）

| 状态 | 📋 Planned |

**动机**：当前 GradientDifferenceLoss 仅使用一阶导数。扩展到二阶（Laplacian）可能进一步提升 smoothness。

**公式**（草案）：

```
L_grad2 = L_grad1 + w_lap * MAE( ∇²pred, ∇²target )
```

---

## 当前实验结果汇总

| 损失组合 | mean_rel_mae | 备注 |
|----------|-------------|------|
| L1 only (no physics) | ~0.646 (v2.1) | 残差学习 baseline |
| L1 + grad (w=0.2) | ~0.622 | 轻微提升 |
| L1 + div (w=0.002, no warmup) | ~0.724 | 无 warmup 反而更差 |
| L1 + div (w=0.002, warmup=0.3) | ~0.640 | warmup 有效 |
| L1 + grad + div (two-stage) | ~0.573 (v2.22) | 当前最佳 |

> 以上数据来自 4-seed 测试或单次运行，详见 `docs/RESULTS.md`。

**关键发现**：
1. 简单的 L1+div 组合在无 warmup 时效果很差（模型过早被物理约束限制）。
2. Warmup 策略有效缓解了 L1+div 的退化问题。
3. 两阶段训练（先只用 L1+grad 训练，再从 checkpoint 加入 div 微调）取得了最佳效果。
4. Adaptive divergence 在稳定性上优于固定权重。

---

## 权重调优指南

### Divergence Loss 权重建议

| 策略 | weight 范围 | 适用场景 |
|------|------------|----------|
| 固定小权重 | 0.0005 – 0.002 | 全程使用，需要 warmup |
| 两阶段 | 0.001 – 0.002 | Stage 2 使用，推荐方案 |
| Adaptive | [0.0, 0.005] | 自动调节，需要设置 ref/tolerance |

### Gradient Loss 权重建议

| 策略 | weight 范围 | 适用场景 |
|------|------------|----------|
| 轻度 | 0.05 – 0.1 | 推荐，v2.22 使用 0.1 |
| 中度 | 0.1 – 0.3 | 速度场 smoothness 优先 |
| 重度 | >0.3 | 不推荐，会损害 pressure |

---

## 添加新物理损失

1. 在 `src/airfoil_cfd_ml_v2/losses.py` 或 `src/airfoil_cfd_ml/losses/` 下实现为 `nn.Module`。
2. 添加对应的 config 字段到 `TrainConfig`。
3. 在 `tests/test_training_smoke.py` 中添加 smoke test。
4. 运行一次小规模实验验证损失值合理（不为 0，不为 NaN，数量级合理）。
5. 更新本文档。

---

*最后更新：2026-06-18*
