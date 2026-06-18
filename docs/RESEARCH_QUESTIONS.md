# Research Questions

本文档定义 AirfoilCFD-ML 项目的五大核心研究问题，每项包含研究动机、实验设计、评估指标和当前状态。

---

## RQ1: Model Inductive Bias — 模型归纳偏置

### 动机

不同神经网络架构对物理场预测任务有不同的归纳偏置：
- CNN/U-Net 擅长捕捉局部空间特征。
- FNO (Fourier Neural Operator) 在傅里叶空间中学习全局算子，理论上更适合求解 PDE。
- Transformer-based 架构通过 attention 机制建模长程依赖。

**核心问题**：在翼型流场预测任务上，不同架构的精度-效率 trade-off 是什么？

### 实验设计

| 模型 | 输入 | 关键超参 | 状态 |
|------|------|----------|------|
| SimpleCNN | (B,3,128,128) | hidden_channels, n_layers | ✅ Baseline |
| DfpNet (U-Net) | (B,3,128,128) | channel_exponent=6, ~7.8M params | ✅ Baseline |
| ResU-Net | (B,3,128,128) | 同上 + residual blocks | 📋 Planned |
| FNO | (B,C,128,128) | modes, width | 📋 Planned |
| Geo-FNO-lite | (B,C+geo,128,128) | modes, width, geo_channels | 📋 Planned |
| Transolver-lite | (B,C,128,128) | n_layers, n_heads, d_model | 📋 Planned |

### 评估指标

- **精度**：mean_rel_mae（所有模型统一评估协议）。
- **效率**：参数量、推理时间 (ms/sample)、训练时间 (GPU hours)。
- **物理一致性**：预测速度场的 divergence（应接近 0）。
- **泛化能力**：不同几何形状/来流条件下的误差分布。

### 当前发现

- DfpNet (U-Net) 作为当前 baseline，两阶段训练后 mean_rel_mae 约 0.573。
- SimpleCNN 精度低于 U-Net，但作为快速验证 baseline 有用。
- 其他模型尚未在统一协议下比较。

### 状态

🚧 In Progress — DfpNet 和 SimpleCNN 已实现。ResU-Net / FNO / Geo-FNO-lite / Transolver-lite 为 📋 Planned。

---

## RQ2: Geometry Encoding — 几何编码

### 动机

当前模型仅通过二值 mask 获取几何信息。更丰富的几何表示可能帮助模型更好地理解翼型形状与流场之间的关系。

**核心问题**：不同几何编码方式（mask / xy-coordinates / SDF / boundary distance）对模型性能的影响有多大？

### 候选编码方案

| 编码方式 | 通道数 | 描述 | 状态 |
|----------|--------|------|------|
| Mask (Binary) | 1 | 当前使用的二值 mask | ✅ Implemented |
| XY Coordinates | 2 | 归一化空间坐标 (x/W, y/H) | 📋 Planned |
| SDF | 1 | 有符号距离场（流体为正，固体为负） | 📋 Planned |
| Boundary Distance | 1 | 到最近壁面的欧氏距离 | 📋 Planned |
| Multi-scale SDF | 2-3 | 多尺度 SDF（不同平滑程度） | 📋 Planned |

### 实验设计

1. 固定模型为 DfpNet (U-Net)，分别用不同几何编码训练。
2. 控制其他变量不变（optimizer, scheduler, loss weights, seed）。
3. 每个配置跑 3-4 个 seed 取均值。

### 评估指标

- mean_rel_mae（主要指标）。
- 壁面附近（mask 边界 ±4px）的局部误差。
- 尾流区域的方向性误差。

### 状态

📋 Planned — 尚未实现。优先实现顺序：XY Coordinates → SDF → Boundary Distance。

详见 [docs/GEOMETRY_ENCODING.md](docs/GEOMETRY_ENCODING.md)。

---

## RQ3: Physics-Aware Training — 物理感知训练

### 动机

纯数据驱动的损失函数（L1/L2）不保证预测结果满足物理约束（如不可压缩流动的散度为零）。引入物理约束损失可能提升预测的物理一致性和泛化能力。

**核心问题**：哪些物理约束损失函数能有效提升预测精度和物理一致性？如何平衡 data loss 和 physics loss 的权重？

### 候选物理损失

| 损失函数 | 物理意义 | 实现复杂度 | 状态 |
|----------|----------|------------|------|
| Divergence Loss | 不可压缩流动 ∇·u = 0 | 低 | ✅ Implemented |
| Gradient Difference Loss | 空间梯度一致性 | 低 | ✅ Implemented |
| Boundary Loss | 壁面无滑移条件 u = 0 | 中 | 📋 Planned |
| Wake Loss | 尾流区域定向加权 | 中 | 📋 Planned |
| Spectral Loss | 频率域能量分布匹配 | 高 | 📋 Planned |
| Multi-order Gradient Loss | 高阶空间导数一致性 | 中 | 📋 Planned |

### 实验设计

1. **消融实验**：逐个添加物理损失，测量精度和物理一致性的变化。
2. **权重扫描**：对每个损失做权重扫描（如 divergence weight ∈ {0.0001, 0.001, 0.01, 0.1}）。
3. **调度策略**：比较 warmup / constant / adaptive 的差异。
4. **组合实验**：找出最优的损失组合。

### 当前发现

- Divergence loss 单独使用时增益有限，在两阶段训练（先无 divergence 训练，再加 divergence fine-tune）中效果最佳。
- Adaptive divergence（基于 validation pressure_rel_mae 自适应调整权重）在稳定性上优于固定权重。
- Gradient difference loss 对速度场预测有一定帮助，但 weight 过大会损害 pressure 预测。

### 状态

🚧 In Progress — Divergence loss 和 Gradient loss 已实现并测试。Boundary / Wake / Spectral loss 为 📋 Planned。

详见 [docs/PHYSICS_LOSSES.md](docs/PHYSICS_LOSSES.md)。

---

## RQ4: Training Acceleration — 训练加速

### 动机

随着模型复杂度增加（FNO, Transformer），训练时间可能成为研究迭代的瓶颈。需要在不大幅修改代码的前提下加速训练。

**核心问题**：AMP / torch.compile / channels_last / LMDB cache / DataLoader 调优能带来多少训练速度提升？精度是否有损失？

### 候选加速策略

| 策略 | 预期加速比 | 精度影响 | 实现复杂度 | 状态 |
|------|------------|----------|------------|------|
| AMP (FP16/BF16) | 1.5–2.5× | 极小 | 低 | 📋 Planned |
| torch.compile | 1.2–1.8× | 无 | 低 | 📋 Planned |
| channels_last (NHWC) | 1.1–1.4× | 无 | 中 | 📋 Planned |
| LMDB Cache | I/O 3–10× | 无 | 中 | 📋 Planned |
| DataLoader 调优 | 1.5–3× | 无 | 低 | 📋 Planned |
| Gradient Accumulation | N/A (增大有效 batch) | 极小 | 低 | 📋 Planned |

### 实验设计

1. 固定 baseline 模型（DfpNet），测量每种加速策略的 wall-clock time。
2. 对比加速前后的精度（确保无退化）。
3. 测量 GPU 利用率和内存占用。
4. 组合最佳策略，测量端到端加速比。

### 状态

📋 Planned — 尚未系统实现。DataLoader 调优部分可配置（num_workers, pin_memory），但未做系统 benchmark。

详见 [docs/TRAINING_ACCELERATION.md](docs/TRAINING_ACCELERATION.md)。

---

## RQ5: Uncertainty and Failure Cases — 不确定性与失败案例

### 动机

代理模型在某些流动条件下可能表现很差（如大攻角分离流）。了解模型的失败模式和预测不确定性对于工程应用至关重要。

**核心问题**：哪些流动条件下代理模型容易失败？如何估计预测不确定性？

### 研究方向

| 方向 | 方法 | 状态 |
|------|------|------|
| Failure Case Analysis | 按几何/来流条件分组分析误差 | 📋 Planned |
| MC Dropout | 训练时启用 Dropout，推理时多次采样估计不确定性 | 📋 Planned |
| Deep Ensemble | 独立训练多个模型，用预测方差估计不确定性 | 📋 Planned |
| Error Map Visualization | 可视化每个样本的空间误差分布 | 📋 Planned |
| Out-of-Distribution Detection | 检测训练分布之外的输入 | 📋 Planned |

### 实验设计

1. **失败案例分析**：
   - 按攻角、翼型厚度、弯度等几何参数分组。
   - 统计每组的 mean_rel_mae。
   - 找出误差最大的前 5% 样本，分析共同特征。
2. **不确定性估计**：
   - 实现 MC Dropout (inference 时保留 Dropout, 采样 30 次)。
   - 实现 Deep Ensemble (4 个独立训练的模型)。
   - 比较两种方法的 uncertainty calibration（预测方差 vs 实际误差的相关性）。

### 状态

📋 Planned — 尚未实现。

详见 [docs/LIMITATIONS.md](docs/LIMITATIONS.md)。

---

## 实验管理规范

所有研究问题的实验均需遵守：

1. **Config 驱动**：每次实验对应一个 YAML config 文件，存放在 `configs/`。
2. **Seed 公平对比**：同一问题下的不同方法使用相同的 seed 集合 {42, 43, 44, 45}。
3. **统一评估协议**：使用 `scripts/evaluate_v2.py` 计算 mean_rel_mae，仅统计流体区域。
4. **结果记录**：实验结果写入 `docs/RESULTS.md`，原始日志存入 `scan_logs/`。
5. **可复现**：checkpoint 保存完整 config 和 normalization stats。

## 研究优先级

基于实现复杂度和预期收益，建议研究优先级：

1. **RQ4 (Training Acceleration)** → 最低风险，立即提升迭代速度。
2. **RQ2 (Geometry Encoding)** → 中等复杂度，可能显著提升精度。
3. **RQ3 (Physics Losses 扩展)** → 在已有 divergence/gradient loss 基础上扩展。
4. **RQ1 (Model Zoo 扩展)** → 高复杂度，需要逐个实现和调参。
5. **RQ5 (Uncertainty)** → 依赖前几项的成熟 pipeline。

---

*最后更新：2026-06-18*
