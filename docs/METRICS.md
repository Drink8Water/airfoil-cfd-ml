# Metrics Reference

本文档定义 AirfoilCFD-ML 项目中所有评估指标的计算公式、物理意义和使用建议。

---

## 指标总览

| 类别 | 指标 | 含义 | Lower is better | 取值范围 |
|------|------|------|:---:|----------|
| Regression | `{ch}_mae` | Mean Absolute Error per channel | ✅ | [0, ∞) |
| Regression | `{ch}_rmse` | Root Mean Square Error per channel | ✅ | [0, ∞) |
| Regression | `{ch}_rel_mae` | Relative MAE per channel | ✅ | [0, ∞) |
| Regression | `mean_rel_mae` | Mean of 3-channel rel_mae | ✅ | [0, ∞) |
| Physics | `divergence_error` | Mean absolute velocity divergence | ✅ | [0, ∞), ideally 0 |
| Physics | `vorticity_error` | Vorticity field MAE vs target | ✅ | [0, ∞) |
| Physics | `boundary_rel_mae` | RelMAE near airfoil surface | ✅ | [0, ∞) |
| Physics | `wake_rel_mae` | RelMAE in wake region | ✅ | [0, ∞) |
| Spectral | `spectral_error` | FFT amplitude MAE | ✅ | [0, ∞) |
| Spectral | `energy_spectrum_error` | Radial energy spectrum relative error | ✅ | [0, ∞) |
| Efficiency | `n_parameters` | Trainable parameter count | — | ℕ |
| Efficiency | `latency_ms_mean` | Mean inference latency (ms) | ✅ | [0, ∞) |

---

## 1. Regression Metrics

所有回归指标仅在流体区域（mask == 1）计算。

**MAE (Mean Absolute Error)** per channel `c`:

```
MAE_c = (1/N_fluid) * Σ |pred_c − target_c|
```

**RMSE (Root Mean Square Error)** per channel `c`:

```
RMSE_c = sqrt( (1/N_fluid) * Σ (pred_c − target_c)² )
```

**Relative MAE** per channel `c`:

```
RelMAE_c = Σ |pred_c − target_c| / Σ |target_c|
```

分母 over all fluid pixels。RelMAE 消除物理量纲差异，可以跨 channel 比较（pressure 量级 ~0–1，速度量级 ~0–200）。

**mean_rel_mae** — 项目的主排序指标：

```
mean_rel_mae = (pressure_rel_mae + u_rel_mae + v_rel_mae) / 3
```

---

## 2. Physics Metrics

### 2.1 divergence_error

```
divergence_error = mean |∂u/∂x + ∂v/∂y|
```

偏导数通过中心有限差分近似。对于不可压缩流动，连续性方程要求 ∇·u = 0。该指标衡量预测速度场的不可压缩性代理——值越低，预测越满足质量守恒。

**注意**：低 divergence + 低 MAE 才是真正的好结果。模型可能通过输出零速度场（divergence=0 但 MAE 巨大）来"作弊"。

### 2.2 vorticity_error

```
ω = ∂v/∂x − ∂u/∂y
vorticity_error = mean |ω_pred − ω_target|
```

涡量是流场结构的关键描述量。翼型升力与环量（涡量面积分）直接相关。该指标衡量预测流场与目标流场在涡结构上的一致性。

### 2.3 boundary_rel_mae

```
boundary_rel_mae = mean_rel_mae within 3px of the fluid/solid boundary
```

边界区域通过形态学膨胀-腐蚀（max-pool 近似）提取后与流体 mask 求交集。翼型壁面附近梯度最大，是代理模型最困难的预测区域之一。几何编码（SDF, boundary distance）预期主要改善此指标。

### 2.4 wake_rel_mae

```
wake_rel_mae = mean_rel_mae in the wake region
```

尾流区域 = 翼型最右侧固体像素以右 30% 域宽范围内的流体像素。尾流包含边界层分离、动量亏损等关键物理信息。

---

## 3. Spectral Metrics

### 3.1 spectral_error

```
spectral_error = (1/3) * Σ_c mean | |FFT(pred_c)| − |FFT(target_c)| |
```

使用 `torch.fft.rfft2` 比较预测与目标在频率域的幅值谱。低 `mean_rel_mae` 但高 `spectral_error` 可能意味着模型在逐点精度上尚可，但未能捕捉高频（小尺度）结构。

### 3.2 energy_spectrum_error

```
E(k) = mean_{|κ| ≈ k} |FFT(field)[κ]|
energy_spectrum_error = mean_k |E_pred(k) − E_target(k)| / E_target(k)
```

通过径向分箱（radial binning）计算各向同性能量谱 E(k)，衡量预测场与目标场在不同空间尺度上的能量分布一致性。

---

## 4. Efficiency Metrics

### 4.1 n_parameters

```
n_parameters = Σ_{p: requires_grad} p.numel()
```

用于精度-效率 trade-off 分析。

### 4.2 latency_ms_mean

```
latency_ms_mean = mean( { t_i | i = 1..repeat } )  [ms]
```

GPU 上使用 `torch.cuda.synchronize()` 确保精确计时。

**延迟的上下文信息**（随 `eval_metrics.json` 一起保存）：

| 字段 | 含义 | 示例 |
|------|------|------|
| `device` | 设备类型 | `"cuda:0"` |
| `device_name` | GPU 型号 | `"NVIDIA GeForce RTX 3060"` |
| `input_shape` | 推理输入 shape | `[1, 6, 128, 128]` |
| `latency_warmup` | 预热次数 | `5` |
| `latency_repeat` | 计时重复次数 | `20` |

> **重要**：没有硬件上下文的 latency 数字无意义。跨实验比较 latency 时必须确认 device_name 和 input_shape 一致。

---

## 5. 指标使用原则

1. **主排序**：`mean_rel_mae`（回归精度）。
2. **物理一致性**：`divergence_error` + `vorticity_error`。
3. **薄弱区域诊断**：`boundary_rel_mae` + `wake_rel_mae`。
4. **频谱一致性**：`spectral_error` + `energy_spectrum_error`。
5. **效率**：`n_parameters` + `latency_ms_mean`（精度-效率 trade-off）。

**反例**：
- 只看 `mean_rel_mae` 不看物理指标 → 可能选出 physics-inconsistent 的模型。
- 只看 latency 不看 device → 数字无意义。
- 把 smoke test 的指标当作 real benchmark → 误导读者。

---

## 6. Smoke vs Real Benchmark

| 属性 | Smoke Benchmark | Real Benchmark |
|------|:---:|:---:|
| 目的 | CI / pipeline 验证 | 研究方法比较 |
| Config 目录 | `configs/experiment/benchmark_smoke/` | `configs/experiment/main_benchmark/` |
| Epochs | 1 | ≥50 |
| Model size | 极小（hidden=8） | 全尺寸（channel_exponent=6） |
| 报告 CSV | `reports/main_benchmark_smoke.csv` | `reports/main_benchmark.csv` |
| README 展示 | ❌ 仅内部 CI | ✅ |

---

*最后更新：2026-06-18*
