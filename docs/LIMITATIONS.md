# Limitations and Future Work

本文档**诚实**记录当前项目的已知局限、失败模式和未来改进方向。不夸大能力，不声称 SOTA。

---

## 已知局限

### 1. 数据集局限

| 局限 | 影响 | 缓解计划 |
|------|------|----------|
| 单一翼型族 | 模型可能无法泛化到不同翼型族（如超临界翼型） | 📋 多数据集评估 |
| 固定分辨率 (128×128) | 无法处理更高/更低分辨率的网格 | 📋 多分辨率测试 |
| 定常流动假设 | 无法预测非定常效应（涡脱落、颤振） | ❌ 超出当前项目范围 |
| 2D 流动 | 无法处理 3D 效应（翼尖涡、展向流动） | ❌ 超出当前项目范围 |
| 训练/测试分布 | 未知训练集与测试集之间的分布差异（几何/来流条件） | 📋 分布分析工具 |
| 数据集来源 | 课程提供的 CFD 数据，开放性和可复现性有限 | 📋 公开数据集迁移 |

**当前数据使用声明（来自 `docs/DATA.md`）**：

> 原始数据集不包含在本仓库中。数据集来源为课程提供的 CFD 样本，用于学术项目实验。

### 2. 模型局限

| 局限 | 影响 | 缓解计划 |
|------|------|----------|
| 单一架构 (U-Net) | 未与其他架构（FNO, Transformer）对比 | 📋 Model Zoo 扩展 |
| 隐式几何编码 | 模型需要从 mask 隐式学习几何，效率可能较低 | 📋 显式几何编码 |
| 无不确定性估计 | 无法判断预测的可靠性 | 📋 MC Dropout / Deep Ensemble |
| 确定性预测 | 无法捕捉 CFD 中的混沌/湍流随机性 | ❌ 需要概率模型 |

### 3. 物理一致性局限

| 局限 | 影响 | 缓解计划 |
|------|------|----------|
| 散度不严格为零 | 即使加入 divergence loss，预测仍不完全满足 ∇·u=0 | 📋 更强的物理约束 |
| 无质量守恒约束 | 预测可能不满足质量守恒 | 📋 可探索 |
| 无量纲化不一致 | 可能需要更仔细的归一化策略 | 📋 归一化消融实验 |

### 4. 工程局限

| 局限 | 影响 | 缓解计划 |
|------|------|----------|
| 无 CI/CD | 无法自动运行测试 | 📋 GitHub Actions |
| 无版本化的数据 pipeline | 数据预处理不可追踪 | 📋 DVC / 数据版本控制 |
| 无 Docker 支持 | 环境复现有平台差异风险 | 📋 Dockerfile |
| 手动实验管理 | 无 ML experiment tracking（如 W&B） | 📋 可选集成 |
| 无 profiling 数据 | 不了解当前训练瓶颈 | 📋 benchmark 脚本 |

---

## 已知失败模式

### 模式 1：大 divergence weight 导致 trivial solution

**现象**：当 divergence_loss_weight 过大时（≥0.01），模型趋于输出零速度场（trivial solution），因为 ∇·0 = 0。

**当前缓解**：
- Warmup scheduling（先让模型学会预测流场）。
- Adaptive divergence（自动调节权重）。
- 两阶段训练（在第二阶段才引入 divergence）。

**待改进**：
- 研究更好的物理约束施加方式（如硬约束/投影方法）。

### 模式 2：Pressure 预测精度显著低于速度

**现象**：在 mean_rel_mae 中，`pressure_rel_mae` 通常为 0.8–1.2，而 `u_rel_mae` 和 `v_rel_mae` 通常为 0.1–0.6。

**可能原因**：
- Pressure 的物理范围远小于速度，但 rel_mae 的归一化方式可能放大误差。
- U-Net 的架构可能更适合速度场预测。
- 当前损失权重（pressure=5.0）可能仍不够。

**待改进**：
- 研究 pressure-specific 的架构修改（如 pressure 专用解码分支）。
- 不同的 pressure 归一化方案。
- Pressure Poisson 物理约束。

### 模式 3：壁面附近误差大

**现象**：即使在流体区域，mask 边界附近的预测误差显著高于远离壁面的区域。

**可能原因**：
- 壁面附近的速度梯度大，更难预测。
- 边界 taper 和 obstacle edge weight 可能过度降低了这些区域的学习信号。
- 网格分辨率不足以捕捉边界层。

**待改进**：
- 更精细的边界处理（如 Boundary Loss）。
- 更高分辨率的边界层区域。
- 自适应网格加密（仅在推理时，不需要重新训练）。

### 模式 4：大攻角/分离流预测差

**现象**（待验证）：在大攻角条件下，流动可能出现分离，代理模型的预测误差可能显著增大。

**待改进**：
- 按来流条件分组的误差分析（需先实现 Failure Case Analysis）。
- 针对分离流的专门训练数据增强。

---

## 未来工作路线图

### Phase 1：基础设施（近期）

- [ ] GitHub Actions CI（自动运行 pytest）。
- [ ] 训练 acceleration benchmark（DataLoader → AMP → compile）。
- [ ] 公共数据集迁移评估（如 AirfRANS、ANYS 等公开 airfoil 数据集）。

### Phase 2：模型与编码（中期）

- [ ] Geometry encoding：XY coords → SDF → Boundary Distance。
- [ ] Model Zoo：ResU-Net → FNO → Geo-FNO-lite → Transolver-lite。
- [ ] 统一 benchmark 框架：同 config、同 seed、同评估协议。

### Phase 3：物理与分析（中远期）

- [ ] Boundary Loss + Wake Loss + Spectral Loss。
- [ ] Failure Case Analysis（按几何/来流条件分组）。
- [ ] Uncertainty Estimation（MC Dropout + Deep Ensemble）。

### Phase 4：生产化（远期）

- [ ] Docker 支持。
- [ ] ONNX/TensorRT 推理优化。
- [ ] 交互式可视化 demo（Gradio/Streamlit）。
- [ ] 论文撰写。

---

## 不做什么（Non-goals）

为保持项目聚焦，以下方向明确不做：

- ❌ **3D 翼型 / 全机 CFD**：超出项目初始范围。
- ❌ **非定常流动**：需要时间序列数据，当前数据集不支持。
- ❌ **与 CFD solver 的耦合**：目标是 surrogate model，不是 solver 替代品。
- ❌ **实时推理**：latency 不是当前优先级。
- ❌ **声称 SOTA**：本项目是研究框架，不声称超越任何特定方法。

---

## 引用与致谢

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{airfoil-cfd-ml,
  author = {Qi Yutian},
  title = {AirfoilCFD-ML: Research-Oriented Scientific Machine Learning for
           Airfoil Flow Surrogate Modeling},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Drink8Water/airfoil-cfd-ml}
}
```

---

*最后更新：2026-06-18*
