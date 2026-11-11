# Aperture-Diffraction-Phase

光阑 → 衍射图样的正向预测，以及「给定目标图样 → 反推光阑」。
核心是把装置里那块相位片/毛玻璃建模成一块**可学习的复振幅屏**，放在已知的衍射传播层前面：

```
E(x') = P_z[ A(x) · T(x) ]              A: 光阑，T = a(x)·e^{iφ(x)}: 屏幕
I(x') = |E(x')|² + B(x')                B: 探测面非相干背景
y     = log1p(100·I/peak) / max        与相机读数同一条压缩管线
```

可学习的只有 `φ`（相位）、`a`（振幅，可选）、`B`（背景，可选）；传播 `P_z` 由波长、距离、
像素确定，**不含任何可训练参数**。所以正向永远是精确物理，反设计时梯度也是真梯度。

## 1. 数据流

```
generate_v2.py ──► data/<name>/ ──► train.py ──► runs/<name>/nf_1/<model>/best_model.pt
                                                  │
                                                  └──► design.py ──► design_runs/<name>/
```

## 2. 模块：generate.py（基础定义，被 generate_v2 复用）

- `PhysicsConfig`：`grid=256`、`input_pitch_m=10e-6`、`wavelength_m=532e-9`、
  `oversample=2`、`pad_factor=2`、`far_threshold=0.1`、`log_alpha=100`。
- `STRATEGIES`：8 类几何光阑（圆孔、方孔、双孔、环、光栅、三角形、十字、字母 F）。
- `make_aperture` / `make_hologram` / `apply_holograms`：自然光阑 + 设计产物风格的密集条纹掩模。
- `support_width(aperture)` → `D`（非零像素包围盒的较长跨度）。
- `geometry(aperture, nf, config)` → `z = D²/(λ·NF)`、`detector_pitch = λz/(N·Δx)`、视场。
- `fraunhofer_raw`：NF < 0.1 用的 FFT 远场。
- `angular_spectrum_raw`：缩放角谱（含精确角谱色散修正），输入复振幅，输出观察网格上的强度。
- `postprocess`：峰值归一化 + `log1p(α·I/peak)` + 除以最大值 → `[0,1]`。
- `split_ids`：按策略分层的 80/10/10 划分。
- `validate`：传播验证（FFT/ASM 一致性、补零收敛、RS1 抽样）。

## 3. 模块：generate_v2.py（生成含屏幕的数据）

- `GlassConfig(correlation_m, rms_rad, shared)`：屏幕统计。预设 `weak / moderate / strong / groundglass`。
- `phase_screen(rng, grid, pitch, correlation_m, rms_rad)`：周期高斯随机场，
  相关长度按自相关 1/e 半径定义；`correlation_m → 0` 时退化为逐像素独立（白噪声）。
- `scaled_field(aperture, z, config)`：复数版缩放角谱。
- `field_after_glass(aperture, phase)` = `A · e^{iφ}`。
- `raw_intensity(field, z, config, method)`：两种标签路径（FFT / ASM）。
- `validate`：`glass_rms`、`glass_correlation`、补零收敛、零相位恒等、屏幕确有影响，
  并画 `phase_check.png`、写 `phase_metrics.csv`。
- `generate`：外层按样本、内层按 NF，写 `apertures.npy / glass_phase.npy / splits.npz /
  nf_*/patterns.npy / nf_*/preview.png / manifest.json`（最后写，`status=complete` 才能训练）。

## 4. 模块：train.py（训练）

### 4.1 三个可学习模块

| 模块 | 参数 | 说明 |
|---|---:|---|
| `LearnedScreen.phase` | 65,536 | `φ(x)`，弧度，初始 0，每步夹到 ±π |
| `LearnedScreen.amplitude` | 65,536 / 256 | `a = sigmoid(α)`；`--amp-grid 16` 时在 16×16 上参数化再双线性放大 |
| `DetectorBackground.logit` | 65,536 | `B = softplus(β) ≥ 0`，加到强度上再压缩 |

### 4.2 固定传播层 `AngularSpectrumDetectorLayer`

无参数，只有 `register_buffer`：非近轴色散修正、传播波掩模、chirp 系数、输出索引。
`raw_intensity(obj, z)` 接受 `[B,1,G,G]` 的**复振幅**，流程是
`repeat×oversample → 补零 → FFT → ×correction → IFFT → ×chirp → FFT → 裁剪 → |·|²×scale²`；
`compress(raw)` 做峰值归一化 + `log1p` + `/max`。

### 4.3 模型组合

| `--models` | 组成 | 参数量 | 用途 |
|---|---|---:|---|
| `phase`（默认） | `LearnedScreen` + 传播层（+ 可选背景） | 65,536~196,608 | 主力，最快 |
| `phase_residual` | 上面的结果 + 零初始化残差 U-Net | +1,928,705 | 吸收离散化误差 |
| `physics` | 只有物理层 + 残差 U-Net（无屏幕） | 1,928,705 | 对照 |

`ConditionedUNet` 只在后两种里用到：3 级编码器（256→128→64→32）、每级两个 `3×3 conv + GroupNorm + GELU`、
3 个转置卷积 + 1 个 `1×1` 输出，感受野约 68 px，`out_conv` 零初始化（从恒等开始学）。

### 4.4 损失

```
L = L1(y_pred, y_true)
  + λ_f · L1( log|FFT(y_pred)| , log|FFT(y_true)| )        λ_f = 0.02
  + λ_s · (1 − SSIM)                                        λ_s = 0
  + λ_r · mean|ΔI|                                          λ_r = 1e-3（仅残差模型）
```

### 4.5 优化

- 屏幕/背景参数用 `--phase-lr 0.02`（`weight_decay=0`），其余用 `--lr 3e-4`；
- AdamW + cosine 退火，每步后把 `φ∈[−π,π]`、`sigmoid(α)` 的 `α∈[−6,6]`、`β∈[−20,5]` 夹住；
- 按验证集 L1 保存 `best_model.pt`；`selection.json` 记录 test 指标与三个屏幕参数的统计。

### 4.6 输出

```
runs/<name>/nf_1/<model>/
    best_model.pt            state_dict 含 screen.phase / screen.amplitude / background.logit
    learned_phase.npy  learned_amplitude.npy  learned_background.npy
    history.csv  selection.json  test_metrics.json  loss_curve.png  train_result.png
```

## 5. 模块：design.py（反设计）

### 5.1 场景

`build_scene(args)`：把目标图样缩放到 `target_size`（默认 96）、四周留 `margin`，
放进 `grid` 大的画布；ROI 由 `--roi-mode auto|center` 决定；`mask` 是 ROI 的指示图。

### 5.2 正向（纯物理）

```
load_phase_model(checkpoint)  →  {layer, screen, background, config, nf}
compose_raw(...)  =  compress( raw_intensity(A·T, z) [+ B] )
predict(...)      =  expm1(y·log1p(α)) / α            → 强度（峰值归一化）
```

`--design-background signal`（默认）只用信号；`total` 会把背景一起拟合（会压平对比度）。

### 5.3 优化与损失

```
L = mean_ROI( |E|/s − T )²             s 每步按 ROI 范数重算（对整体亮度不敏感）
  + 0.02 · Σ_outside(|E|/s)² / N_roi
  + 0.002 · mean(a(1−a))               二值化
  + 2.0 · (mean(a) − 0.4)²             平均透过率
```

`--modulation binary` 前 55% 步用软掩模、之后硬 0/1 + STE；`--modulation phase` 用 `e^{iθ}`（复振幅正向支持）。

### 5.4 输出

```
design_runs/<name>/
    aperture.npy / aperture.png  pattern_pred.npy  pattern_fit.png  pattern_roi.png  target_roi.png
    comparison.png   ① 目标 ② φ̂ 正向 ③ 真实相位复核 ④ 两者之差 ⑤ 光阑 ⑥ 全视场 ⑦ 全视场(真实) ⑧ 收敛
    result.json
```

`--true-phase` 只在仿真里给，用于画 ③ 和存 `pattern_true_intensity.npy`，**不参与优化**。

## 6. 目录

```text
Aperture-Diffraction-Phase/
├── README.md
├── instructions.txt        命令速查
├── .gitignore
├── generate.py             光阑/几何/传播/后处理（基础定义）
├── generate_v2.py          含屏幕的数据生成
├── train.py                可学习屏幕 + 固定传播层（+ 可选残差）
├── design.py               目标图样 → 光阑
├── data/  runs/  design_runs/  inputs/
```

`data/<name>/`：`apertures.npy`、`glass_phase.npy`、`splits.npz`、`manifest.json`、
`validation.json`、`nf_*/patterns.npy`、`nf_*/preview.png`。

## 7. 参数量一览

| 配置 | 参数量 |
|---|---:|
| `--models phase` | 65,536 |
| `+ --screen-mode complex`（`--amp-grid 16`） | 65,792 |
| `+ --background map` | 131,072 |
| 全部（复振幅 + 背景） | 131,328 |
| `--models phase_residual` | 1,994,241 |
| `--models physics` | 1,928,705 |

## 8. 指标量级

| 数据 | 模型 | test MAE |
|---|---|---:|
| 相位屏 rms 1.5 rad | `phase` | 3.8e-5 |
| 相位屏 rms 1.5 + 振幅 15% + 背景 3% | `phase` | 0.151 |
| 同上 | `phase --background map` | 0.0185 |
| 同上 | `phase --screen-mode complex --background map` | 0.0142 |

判读：`test.mae` 到 1e-5 量级 = 抓到了；1e-3 = 有非相位偏差或陷局部极小；1e-2 以上 = 模型形式不成立。
