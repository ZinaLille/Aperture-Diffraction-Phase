"""训练：把干扰建模成一块**可学习的相位屏 φ̂**，放进物理层一起训练。

    phase          = 可训练相位屏 φ̂（256×256，初始 0）+ 固定传播层，纯物理
    phase_residual = 上面的结果 + 零初始化残差 U-Net（吸收离散化误差，可选）
    physics        = 不带相位屏的老物理层（对照组，等价于 φ̂≡0）

损失 = L1 + λ_f·log 频谱 L1 + λ_s·(1-SSIM) + λ_r·|ΔI|

φ̂ 只有 65k 个参数，而 808 个样本 × 65536 像素 = 5.3×10^7 个观测，严重超定；
训练好之后 best_model.pt 里的相位就是这块玻璃（也可导出 learned_phase.npy）。
design.py 直接用它做纯物理正向，不再有"网络近似"造成的代理误差。

用法：
    python train.py --data data/v2_nf1_thick --groups 1 --propagator hybrid \
        --lambda-freq 0.02 --epochs 60 --out runs/v2_nf1_thick
    python train.py --data data/v2_nf1_thick --models phase phase_residual --out runs/cmp
    python train.py --data data/v1 --models physics --out runs/v1_ref

每个模型目录输出：
    best_model.pt  learned_phase.npy  history.csv  selection.json
    test_metrics.json  loss_curve.png  train_result.png
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import PhysicsConfig, image_metrics, support_width


# ============================================================
# 1. 网络与损失
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


def ssim(pred, target, window=11, sigma=1.5):
    """内置可微 SSIM；输入 (B,1,H,W) 已归一化到 [0,1]。

    使用高斯窗、非重叠填充，返回 batch 平均值。尺寸小于 11 时自动缩小窗。
    """
    height, width = pred.shape[-2:]
    window = min(window, height, width)
    if window % 2 == 0:
        window -= 1
    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - window // 2
    gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
    kernel = kernel.expand(pred.shape[1], 1, window, window).contiguous()

    mu_p = F.conv2d(pred, kernel, padding=window // 2, groups=pred.shape[1])
    mu_t = F.conv2d(target, kernel, padding=window // 2, groups=target.shape[1])
    var_p = (F.conv2d(pred * pred, kernel, padding=window // 2,
                      groups=pred.shape[1]) - mu_p**2).clamp_min(0)
    var_t = (F.conv2d(target * target, kernel, padding=window // 2,
                      groups=target.shape[1]) - mu_t**2).clamp_min(0)
    cov_pt = F.conv2d(pred * target, kernel, padding=window // 2,
                      groups=pred.shape[1]) - mu_p * mu_t
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_p * mu_t + c1) * (2 * cov_pt + c2) /
             ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)))
    return value.mean()


class Loss(nn.Module):
    """L = L1 + λ_f·L1(log|FFT|) + λ_s·(1-SSIM) + λ_r·|ΔI|。"""

    def __init__(self, lambda_freq=0.1, lambda_ssim=0.0, lambda_res=1e-3,
                 use_log_freq=True):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.lambda_ssim = lambda_ssim
        self.lambda_res = lambda_res
        self.use_log_freq = use_log_freq
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, delta=None):
        loss_data = self.l1(pred, target)
        if self.lambda_freq > 0:
            a_pred = torch.abs(torch.fft.fft2(pred.squeeze(1)))
            a_target = torch.abs(torch.fft.fft2(target.squeeze(1)))
            if self.use_log_freq:
                a_pred, a_target = torch.log1p(a_pred), torch.log1p(a_target)
            loss_freq = torch.mean(torch.abs(a_pred - a_target))
        else:
            loss_freq = pred.new_zeros(())
        loss_ssim = (1.0 - ssim(pred, target)) if self.lambda_ssim > 0 else pred.new_zeros(())
        loss_res = (torch.mean(torch.abs(delta))
                    if self.lambda_res > 0 and delta is not None else pred.new_zeros(()))
        total = (loss_data + self.lambda_freq * loss_freq
                 + self.lambda_ssim * loss_ssim + self.lambda_res * loss_res)
        stats = {"l1": loss_data.item(), "freq": loss_freq.item(),
                 "ssim": loss_ssim.item(), "res": loss_res.item()}
        return total, stats


class ConditionedUNet(nn.Module):
    def __init__(self, in_channels, base_ch=32, residual=False, target_mean=0.0):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)
        if residual:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)
        else:
            nn.init.normal_(self.out_conv.weight, 0, 1e-3)
            nn.init.constant_(self.out_conv.bias, target_mean)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.bottleneck(self.pool(e3))
        x = self.dec3(torch.cat((self.up3(x), e3), 1))
        x = self.dec2(torch.cat((self.up2(x), e2), 1))
        x = self.dec1(torch.cat((self.up1(x), e1), 1))
        return self.out_conv(x)


class FraunhoferDetectorLayer(nn.Module):
    """固定夫琅禾费层：FFT → 模平方 → 峰值归一化 → log 压缩。"""

    def __init__(self, alpha=100.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, aperture, z_m=None):
        field = torch.fft.fft2(aperture.squeeze(1))
        intensity = torch.fft.fftshift(field, dim=(-2, -1)).abs().square()
        intensity = intensity / intensity.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        intensity = torch.log1p(self.alpha * intensity)
        intensity = intensity / intensity.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        return intensity.unsqueeze(1)


class AngularSpectrumDetectorLayer(nn.Module):
    """固定可微缩放角谱层；无参数，接收以米为单位的 z。"""

    def __init__(self, config, precision="float32"):
        super().__init__()
        self.config = config
        if precision not in ("float32", "float64"):
            raise ValueError("ASM 精度必须是 float32 或 float64")
        self.real_dtype = torch.float64 if precision == "float64" else torch.float32
        self.complex_dtype = torch.complex128 if precision == "float64" else torch.complex64
        self.alpha = config.log_alpha
        c, pad, n = config.oversample, config.pad_factor, config.grid
        self.fine_grid = n * c
        self.padded_grid = m = self.fine_grid * pad
        dx = config.input_pitch_m / c
        frequency = torch.fft.fftfreq(m, d=dx, dtype=torch.float64)
        q = config.wavelength_m**2 * (frequency[:, None]**2 + frequency[None, :]**2)
        root = torch.sqrt(torch.clamp(1 - q, min=0))
        coefficient = -(2 * torch.pi / config.wavelength_m) * q.square() / (2 * (1 + root).square())
        coord = (torch.arange(m, dtype=torch.float64) - m // 2 - (c - 1) / 2) * dx
        self.register_buffer("correction_coefficient", coefficient, persistent=False)
        self.register_buffer("propagating", q < 1, persistent=False)
        self.register_buffer("chirp_coefficient",
                             torch.pi * coord.square() / config.wavelength_m, persistent=False)
        self.register_buffer("output_indices",
                             m // 2 + (torch.arange(n) - n // 2) * pad, persistent=False)

    def raw_intensity(self, aperture, z_m):
        if aperture.ndim != 4 or aperture.shape[1:] != (1, self.config.grid, self.config.grid):
            raise ValueError("ASM 输入必须是 [B,1,grid,grid]")
        if z_m is None:
            raise ValueError("ASM 需要以米为单位的 z")
        z = torch.as_tensor(z_m, dtype=torch.float64, device=aperture.device).reshape(-1)
        if z.numel() != len(aperture) or not torch.isfinite(z).all() or (z <= 0).any():
            raise ValueError("每个样本需要一个有限正传播距离")
        c = self.config.oversample
        # 支持复振幅输入（光阑 × e^{iφ̂}）：实部/虚部一起传播
        complex_input = torch.is_complex(aperture)
        fine = aperture[:, 0] if complex_input else aperture[:, 0].to(self.real_dtype)
        fine = fine.repeat_interleave(c, -2).repeat_interleave(c, -1)
        offset = (self.padded_grid - self.fine_grid) // 2
        fine = F.pad(fine, (offset, offset, offset, offset))
        spectrum = torch.fft.fft2(torch.fft.ifftshift(fine, dim=(-2, -1)))
        phase = z[:, None, None] * self.correction_coefficient
        correction = torch.polar(torch.ones_like(phase), phase).to(self.complex_dtype)
        corrected = torch.fft.fftshift(
            torch.fft.ifft2(spectrum * (correction * self.propagating)), dim=(-2, -1))
        phase = self.chirp_coefficient[None, :] / z[:, None]
        chirp = torch.polar(torch.ones_like(phase), phase).to(self.complex_dtype)
        field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
            corrected * chirp[:, :, None] * chirp[:, None, :], dim=(-2, -1))), dim=(-2, -1))
        field = field.index_select(-2, self.output_indices).index_select(-1, self.output_indices)
        scale = (self.config.input_pitch_m / c)**2 / (self.config.wavelength_m * z)
        return (field.abs().square() * scale[:, None, None].to(self.real_dtype).square()).unsqueeze(1)

    def compress(self, raw):
        """原始强度 -> 峰值归一化 + log1p + /max（与 generate 的 postprocess 同一条管线）。"""
        value = raw / raw.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        value = torch.log1p(self.alpha * value)
        return (value / value.amax((-2, -1), keepdim=True).clamp_min(1e-30)).to(self.real_dtype)

    def forward(self, aperture, z_m):
        return self.compress(self.raw_intensity(aperture, z_m))


class LearnedScreen(nn.Module):
    """可学习的复振幅屏 T = a·e^{iφ}。一块固定的玻璃/相位片 = 一个 256×256 的张量。

    - 纯相位（默认）：只有 φ，|T| = 1；
    - 复振幅（--screen-mode complex）：再加一个吸收/照明不均匀的振幅 a = sigmoid(α)。
    相位是 2π 周期的，每步后夹到 ±π；振幅夹在 [0.0025, 0.9975]。
    """

    def __init__(self, grid, init="zero", complex_screen=False, coarse=0):
        super().__init__()
        if init == "zero":
            values = torch.zeros(1, 1, grid, grid)
        elif init == "random":
            values = torch.empty(1, 1, grid, grid).uniform_(-torch.pi, torch.pi)
        else:
            raise ValueError("phase_init 只能是 zero / random")
        self.phase = nn.Parameter(values)
        self.grid = grid
        self.coarse = int(coarse)
        # 复振幅屏的振幅：
        #   coarse=0     全分辨率，每个像素一个 a（自由度高、容易把残差吸进来）
        #   coarse=16    在 16×16 上参数化再双线性放大（照明/吸收本来就应该平滑）
        self.amplitude = (nn.Parameter(torch.full((1, 1, grid, grid), 4.0))
                          if complex_screen and self.coarse <= 0 else None)
        self.amp_coarse = (nn.Parameter(torch.zeros(1, 1, self.coarse, self.coarse))
                           if complex_screen and self.coarse > 0 else None)

    def amplitude_map(self):
        """返回 (1,1,grid,grid) 的振幅 a；纯相位屏返回 None。"""
        if self.amp_coarse is not None:
            upsampled = F.interpolate(self.amp_coarse, size=(self.grid, self.grid),
                                      mode="bilinear", align_corners=False)
            return torch.sigmoid(4.0 + upsampled)
        if self.amplitude is not None:
            return torch.sigmoid(self.amplitude)
        return None

    def transmission(self, like):
        """复振幅 T（广播到 like 的形状 [B,1,grid,grid]）。"""
        amp = self.amplitude_map()
        if amp is None:
            return torch.polar(torch.ones_like(like), self.phase)
        return torch.polar(amp, self.phase)


class DetectorBackground(nn.Module):
    """探测面上的非相干背景 B(x)（环境杂光/暗电流），加到强度上再压缩。

    用 softplus 保证非负；初值很小（≈4.5e-5），训练中按需长出来。
    单位和传播得到的强度一致，所以不同光阑的"信噪比"会自动不同。
    """

    def __init__(self, grid, init=-10.0):
        super().__init__()
        self.logit = nn.Parameter(torch.full((1, 1, grid, grid), float(init)))

    def forward(self):
        return F.softplus(self.logit)


class ForwardModel(nn.Module):
    def __init__(self, kind, base_ch=32, alpha=100.0, target_mean=0.0,
                 propagator="fraunhofer", physics_config=None, asm_precision="float32",
                 phase_init="zero", screen_mode="phase", background="none", amp_grid=0):
        super().__init__()
        if kind not in ("physics", "phase", "phase_residual"):
            raise ValueError(kind)
        if propagator not in ("fraunhofer", "asm"):
            raise ValueError(propagator)
        if kind.startswith("phase") and propagator != "asm":
            raise ValueError("可学习相位屏只能配合 --propagator asm（或 hybrid 且该组用 ASM）")
        self.kind = kind
        self.propagator = propagator
        self.has_phase = kind.startswith("phase")
        self.has_residual = kind == "phase_residual"
        self.screen_mode = screen_mode
        self.background_mode = background
        config = physics_config or PhysicsConfig(log_alpha=alpha)
        self.physics = (AngularSpectrumDetectorLayer(config, asm_precision)
                        if propagator == "asm" and kind != "baseline"
                        else FraunhoferDetectorLayer(alpha))
        self.screen = (LearnedScreen(config.grid, phase_init, screen_mode == "complex",
                                     coarse=amp_grid) if self.has_phase else None)
        self.background = (DetectorBackground(config.grid) if background == "map" else None)
        if self.has_residual or kind == "physics":
            self.network = ConditionedUNet(4, base_ch, residual=True, target_mean=target_mean)
        else:
            self.network = None

    def forward(self, aperture, condition, z_m=None):
        if self.kind == "baseline":
            maps = condition[:, :, None, None].expand(-1, -1, *aperture.shape[-2:])
            return self.network(torch.cat((aperture, maps), 1)), None, None
        if self.has_phase:
            # 物面复振幅 A·T → 固定传播 → 强度 →（可选）加背景 → 压缩
            obj = aperture.to(torch.complex64) * self.screen.transmission(aperture)
            raw = self.physics.raw_intensity(obj, z_m)
            if self.background is not None:
                raw = raw + self.background()
            simulated = self.physics.compress(raw)
        else:
            simulated = self.physics(aperture, z_m)
        if self.network is None:
            return simulated, simulated, None
        maps = condition[:, :, None, None].expand(-1, -1, *aperture.shape[-2:])
        delta = self.network(torch.cat((aperture, simulated, maps), 1))
        return simulated + delta, simulated, delta

    def step_done(self):
        """优化器 step 之后调用：相位夹回 ±π，振幅夹在 [0.0025, 0.9975]。"""
        if not self.has_phase:
            return
        with torch.no_grad():
            self.screen.phase.clamp_(-torch.pi, torch.pi)
            if self.screen.amplitude is not None:
                self.screen.amplitude.clamp_(-6.0, 6.0)
            if self.screen.amp_coarse is not None:
                self.screen.amp_coarse.clamp_(-4.0, 4.0)
            if self.background is not None:
                self.background.logit.clamp_(-20.0, 5.0)


# ============================================================
# 2. 数据读取
# ============================================================
_GEOMETRY_CACHE = {}


def geometry_arrays(root, group):
    """按组缓存 z 与观察面像素间距；由光阑和 NF 确定性计算。"""
    key = (str(Path(root).resolve()), group)
    if key not in _GEOMETRY_CACHE:
        manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
        config = PhysicsConfig(**manifest["config"])
        nf = next(g["nf"] for g in manifest["groups"] if g["name"] == group)
        apertures = np.load(Path(root) / "apertures.npy", mmap_mode="r")
        z = np.empty(len(apertures), dtype=np.float64)
        pitch = np.empty(len(apertures), dtype=np.float64)
        for i, aperture in enumerate(apertures):
            d = support_width(np.asarray(aperture), config.input_pitch_m)
            z[i] = d * d / (config.wavelength_m * nf)
            pitch[i] = config.wavelength_m * z[i] / (config.grid * config.input_pitch_m)
        _GEOMETRY_CACHE[key] = (z, pitch)
    return _GEOMETRY_CACHE[key]


class GroupData(Dataset):
    """一个 NF 组的一个划分；条件为标准化后的 log(z), log(像素间距)。"""

    def __init__(self, root, group, split):
        self.root, self.group, self.split = Path(root), group, split
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError("数据集尚未完整生成")
        self.config = PhysicsConfig(**self.manifest["config"])
        self.nf = next(g["nf"] for g in self.manifest["groups"] if g["name"] == group)
        self.apertures = np.load(self.root / "apertures.npy", mmap_mode="r")
        self.patterns = np.load(self.root / group / "patterns.npy", mmap_mode="r")
        with np.load(self.root / "splits.npz") as splits:
            self.ids = splits[split].copy()
            train_ids = splits["train"].copy()
        expected = (self.manifest["samples"], self.config.grid, self.config.grid)
        if self.apertures.shape != expected or self.patterns.shape != expected:
            raise ValueError("数据形状与 manifest 不一致")
        self.z_all, self.pitch_all = geometry_arrays(self.root, group)
        self.conditions = np.log(np.stack((self.z_all, self.pitch_all), axis=1))
        self.condition_mean = self.conditions[train_ids].mean(0)
        self.condition_std = np.maximum(self.conditions[train_ids].std(0), 1e-6)
        self._target_mean = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = int(self.ids[index])
        a = torch.from_numpy(np.array(self.apertures[i], copy=True)).unsqueeze(0)
        target = torch.from_numpy(np.array(self.patterns[i], copy=True)).unsqueeze(0)
        condition = torch.tensor(
            (self.conditions[i] - self.condition_mean) / self.condition_std,
            dtype=torch.float32)
        return a, target, condition, i

    def target_mean(self):
        # 只用训练划分的标签，避免验证/测试信息进入输出层初始化。
        if self._target_mean is None:
            train_ids = self.ids
            self._target_mean = float(np.mean(
                [self.patterns[int(i)].mean(dtype=np.float64) for i in train_ids]))
        return self._target_mean

    def z_tensor(self, ids, device):
        return torch.tensor([self.z_all[int(i)] for i in ids],
                            dtype=torch.float64, device=device)


# ============================================================
# 3. 训练与评估
# ============================================================
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用的 CUDA")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ("mae", "rmse", "psnr_db", "corr")}


def epoch_pass(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    totals = {k: 0.0 for k in ("total", "l1", "freq", "ssim", "res")}
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for a, target, condition, ids in loader:
            a, target, condition = a.to(device), target.to(device), condition.to(device)
            z = loader.dataset.z_tensor(ids, device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            pred, _, delta = model(a, condition, z)
            loss, stats = criterion(pred, target, delta)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失出现非有限值")
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.step_done()
            totals["total"] += loss.item() * len(a)
            for key, value in stats.items():
                totals[key] += value * len(a)
            count += len(a)
    return {k: v / count for k, v in totals.items()}


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    rows = []
    for a, target, condition, ids in loader:
        pred = model(a.to(device), condition.to(device), loader.dataset.z_tensor(ids, device))[0]
        for ident, p, t in zip(ids.tolist(), pred[:, 0].cpu().numpy(), target[:, 0].numpy()):
            rows.append({"id": ident, **image_metrics(p, t)})
    return aggregate(rows)


def checkpoint_payload(model, args, dataset, kind, propagator, epoch, val_l1, nf):
    screen = getattr(model, "screen", None)
    amplitude = None
    if screen is not None and screen.amplitude_map() is not None:
        amplitude = screen.amplitude_map().detach().cpu()
    background = (model.background().detach().cpu()
                  if getattr(model, "background", None) is not None else None)
    return {"state_dict": model.state_dict(), "kind": kind, "base_ch": args.base_ch,
            "nf": nf, "epoch": epoch, "val_l1": val_l1,
            "condition_mean": dataset.condition_mean.tolist(),
            "condition_std": dataset.condition_std.tolist(),
            "physics_config": dataset.config.to_dict(),
            "propagator": propagator, "asm_precision": "float32",
            # 相位在 state_dict 的 "screen.phase" 里；振幅/背景再存一份"生效值"，
            # 这样 design.py 不用关心它是全分辨率还是 16×16 参数化的
            "amplitude_map": amplitude, "background_map": background,
            "amp_grid": args.amp_grid,
            "has_phase": bool(getattr(model, "has_phase", False)),
            "has_residual": bool(getattr(model, "has_residual", False)),
            "phase_init": args.phase_init, "phase_lr": args.phase_lr,
            "screen_mode": getattr(model, "screen_mode", "phase"),
            "background_mode": getattr(model, "background_mode", "none")}


def plot_history(history, path, args, kind, nf, initial_l1):
    panels = [("l1", "L1 loss", "L1")]
    if args.lambda_freq > 0:
        panels.append(("freq", "Freq log-magnitude L1", "freq"))
    if args.lambda_ssim > 0:
        panels.append(("ssim", "SSIM loss", "1 - SSIM"))
    if kind in ("physics", "phase_residual") and args.lambda_res > 0:
        panels.append(("res", "Residual regularization", "|ΔI|"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    epochs = [row["epoch"] for row in history]
    for ax, (key, title, ylabel) in zip(axes[0], panels):
        for split in ("train", "val"):
            ax.semilogy(epochs, [max(row[f"{split}_{key}"], 1e-12) for row in history],
                        label=split, lw=1.5)
        if key == "l1":
            ax.axhline(max(initial_l1, 1e-12), ls=":", color="gray", label="initial val")
        ax.set(xlabel="epoch", ylabel=ylabel, title=title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"NF={nf:g}  {kind}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_samples(model, dataset, device, path, kind, nf, n_show=4):
    """每行：光阑 | 物理层 |（残差版才有）预测 | 真值 | 残差/误差。"""
    model.eval()
    rng = np.random.RandomState(20261021)
    count = min(n_show, len(dataset))
    indices = rng.choice(len(dataset), count, replace=False)
    has_residual = kind in ("physics", "phase_residual")
    columns = 5 if has_residual else 4
    fig, axes = plt.subplots(count, columns, figsize=(4 * columns, 4 * count), squeeze=False)
    for row, index in enumerate(indices):
        a, target, condition, ident = dataset[int(index)]
        a = a.unsqueeze(0).to(device)
        condition = condition.unsqueeze(0).to(device)
        z = dataset.z_tensor([int(ident)], device)
        target_np = target[0].numpy()
        predicted, simulated, delta = model(a, condition, z)
        predicted_np = predicted[0, 0].cpu().numpy()
        simulated_np = simulated[0, 0].cpu().numpy()
        if has_residual:
            values = [a[0, 0].cpu().numpy(), simulated_np, predicted_np, target_np,
                      delta[0, 0].cpu().numpy()]
            titles = ["Aperture", "I_sim (physics)", "I_pred (physics+res)",
                      "Ground truth", "Residual ΔI"]
            note = (f"sim MAE={np.mean(abs(simulated_np - target_np)):.5f}  "
                    f"pred MAE={np.mean(abs(predicted_np - target_np)):.5f}")
        else:
            values = [a[0, 0].cpu().numpy(), simulated_np, target_np,
                      np.abs(simulated_np - target_np)]
            titles = ["Aperture", "I_sim (physics + φ̂)", "Ground truth", "Absolute error"]
            note = f"sim MAE={np.mean(abs(simulated_np - target_np)):.5f}"
        for col, (value, title) in enumerate(zip(values, titles)):
            ax = axes[row, col]
            if col == 0:
                image = ax.imshow(value, cmap="gray", vmin=0, vmax=1)
            elif title == "Residual ΔI":
                limit = float(np.abs(value).max()) + 1e-12
                image = ax.imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit)
            elif title == "Absolute error":
                image = ax.imshow(value, cmap="inferno")
            else:
                image = ax.imshow(value, cmap="inferno", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(title, fontsize=12)
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        axes[row, -1].set_xlabel(note, fontsize=9)
    fig.suptitle(f"NF={nf:g} | validation samples", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_group(root, group, args, device, out_root):
    name, nf = group["name"], group["nf"]
    sets = {split: GroupData(root, name, split) for split in ("train", "val", "test")}
    loaders = {split: DataLoader(value, batch_size=args.batch_size,
                                 shuffle=(split == "train"),
                                 generator=torch.Generator().manual_seed(args.seed) if split == "train" else None,
                                 num_workers=args.workers)
               for split, value in sets.items()}
    propagator = ("fraunhofer" if nf < sets["train"].config.far_threshold else "asm") \
        if args.propagator == "hybrid" else args.propagator
    out = out_root / name
    out.mkdir()

    rows = []
    for kind in args.models:
        started = time.perf_counter()
        folder = out / kind
        folder.mkdir()
        seed_all(args.seed)
        model = ForwardModel(kind, args.base_ch, sets["train"].config.log_alpha,
                             sets["train"].target_mean(), propagator,
                             sets["train"].config, phase_init=args.phase_init,
                             screen_mode=args.screen_mode,
                             background=args.background,
                             amp_grid=args.amp_grid).to(device)
        criterion = Loss(lambda_freq=args.lambda_freq, lambda_ssim=args.lambda_ssim,
                         lambda_res=args.lambda_res if kind in ("physics", "phase_residual")
                         else 0.0)
        # 屏参数用单独的（更大的）学习率；不给它们 weight decay，避免被拉回初值
        screen_params = [p for n, p in model.named_parameters()
                         if n.startswith("screen.") or n.startswith("background.")]
        other_params = [p for n, p in model.named_parameters()
                        if not (n.startswith("screen.") or n.startswith("background."))]
        groups = []
        if screen_params:
            groups.append({"params": screen_params, "lr": args.phase_lr, "weight_decay": 0.0})
        if other_params:
            groups.append({"params": other_params, "lr": args.lr, "weight_decay": 1e-4})
        optimizer = torch.optim.AdamW(groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        initial = epoch_pass(model, loaders["val"], criterion, device)
        best = initial["l1"]
        torch.save(checkpoint_payload(model, args, sets["train"], kind, propagator,
                                      0, best, nf), folder / "best_model.pt")
        print(f"NF={nf:g} {kind} epoch 0: val_loss={best:.4e}", flush=True)

        history = []
        for epoch in range(args.epochs):
            train_stats = epoch_pass(model, loaders["train"], criterion, device, optimizer)
            val_stats = epoch_pass(model, loaders["val"], criterion, device)
            scheduler.step()
            row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                   **{"train_" + k: v for k, v in train_stats.items()},
                   **{"val_" + k: v for k, v in val_stats.items()}}
            history.append(row)
            write_csv(folder / "history.csv", history)
            if val_stats["l1"] < best:
                best = val_stats["l1"]
                torch.save(checkpoint_payload(model, args, sets["train"], kind,
                                              propagator, epoch + 1, best, nf),
                           folder / "best_model.pt")
            print(f"NF={nf:g} {kind} epoch {epoch + 1}/{args.epochs}: "
                  f"train_loss={train_stats['l1']:.4e} val_loss={val_stats['l1']:.4e}",
                  flush=True)

        checkpoint = torch.load(folder / "best_model.pt", map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        test = score(model, loaders["test"], device)
        learned, amplitude, background = None, None, None
        if model.has_phase:
            learned = model.screen.phase.detach().cpu().numpy()[0, 0]
            np.save(folder / "learned_phase.npy", learned.astype(np.float32))
            if model.screen.amplitude_map() is not None:
                amplitude = model.screen.amplitude_map().detach().cpu().numpy()[0, 0]
                np.save(folder / "learned_amplitude.npy", amplitude.astype(np.float32))
            if model.background is not None:
                background = model.background().detach().cpu().numpy()[0, 0]
                np.save(folder / "learned_background.npy", background.astype(np.float32))
        with (folder / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test, f, ensure_ascii=False, indent=2)
        with (folder / "selection.json").open("w", encoding="utf-8") as f:
            json.dump({"best_epoch": checkpoint["epoch"],
                       "best_validation_l1": checkpoint["val_l1"],
                       "initial_validation_l1": initial["l1"],
                       "test": test,
                       "screen_mode": args.screen_mode,
                       "background_mode": args.background,
                       "phase": ({"file": "learned_phase.npy",
                                  "rms_rad": float(learned.std()),
                                  "min_rad": float(learned.min()),
                                  "max_rad": float(learned.max())}
                                 if learned is not None else None),
                       "amplitude": ({"file": "learned_amplitude.npy",
                                      "mean": float(amplitude.mean()),
                                      "min": float(amplitude.min()),
                                      "max": float(amplitude.max()),
                                      "std": float(amplitude.std())}
                                     if amplitude is not None else None),
                       "background": ({"file": "learned_background.npy",
                                       "mean": float(background.mean()),
                                       "max": float(background.max())}
                                      if background is not None else None),
                       "wall_seconds": time.perf_counter() - started},
                      f, ensure_ascii=False, indent=2)
        plot_history(history, folder / "loss_curve.png", args, kind, nf, initial["l1"])
        plot_samples(model, sets["val"], device, folder / "train_result.png",
                     kind, nf)
        rows.append({"nf": nf, "model": kind, "propagator": propagator,
                     "best_epoch": checkpoint["epoch"],
                     "best_val_l1": checkpoint["val_l1"], **test})
        print(f"NF={nf:g} {kind} 完成：best val={best:.4e}, test MAE={test['mae']:.4e} | {folder}",
              flush=True)
        if learned is not None:
            extra = ""
            if amplitude is not None:
                extra += f"；振幅 a: {amplitude.min():.2f}~{amplitude.max():.2f}"
            if background is not None:
                extra += f"；背景 B: 均值 {background.mean():.2e}"
            print(f"  屏幕：相位 rms={learned.std():.3f} rad{extra}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/v1", help="generate.py 的输出目录")
    parser.add_argument("--out", default=None, help="新目录；默认 runs/<时间戳>")
    parser.add_argument("--groups", type=float, nargs="+", help="只训练指定 NF 组")
    parser.add_argument("--models", nargs="+", choices=("phase", "phase_residual", "physics"),
                        default=("phase",),
                        help="phase=可学习相位屏+纯物理（推荐）；phase_residual=再叠残差 U-Net；"
                             "physics=不带相位屏的老物理层对照")
    parser.add_argument("--propagator", choices=("fraunhofer", "asm", "hybrid"),
                        default="hybrid",
                        help="物理层；hybrid 在 NF<切换点用 FFT，其余用 ASM")
    parser.add_argument("--phase-lr", type=float, default=0.02,
                        help="相位屏的学习率（比网络大一个数量级）")
    parser.add_argument("--phase-init", choices=("zero", "random"), default="zero",
                        help="相位屏初始化：zero = 从无玻璃开始（最稳），random = 均匀随机")
    parser.add_argument("--screen-mode", choices=("phase", "complex"), default="phase",
                        help="phase = 纯相位屏（|T|=1）；complex = 复振幅屏 a·e^{iφ}，"
                             "额外拟合吸收/照明不均匀")
    parser.add_argument("--background", choices=("none", "map"), default="none",
                        help="map = 再拟合一张非负的探测面背景 B(x)（环境杂光/暗电流）")
    parser.add_argument("--amp-grid", type=int, default=16,
                        help="复振幅屏里振幅的参数化粒度：16 = 在 16×16 上参数化再放大（平滑、推荐）；"
                             "0 = 全分辨率逐像素（自由度高，容易把残差吸进来）")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-freq", type=float, default=0.02,
                        help="频谱 L1 权重；有玻璃的数据用 0.02（太大会压住相位屏的学习）")
    parser.add_argument("--lambda-ssim", type=float, default=0.0,
                        help="内置 SSIM，无需额外依赖")
    parser.add_argument("--lambda-res", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0, help="Windows 下建议 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.base_ch < 8 or args.base_ch % 8:
        parser.error("epochs/batch-size 必须为正，base-ch 必须是 8 的倍数")
    if args.lr <= 0 or args.phase_lr <= 0 or min(args.lambda_freq, args.lambda_ssim,
                                                args.lambda_res) < 0:
        parser.error("学习率必须为正，损失系数不能为负")
    needs_asm = any(kind.startswith("phase") for kind in args.models)
    if needs_asm and args.propagator == "fraunhofer":
        parser.error("phase / phase_residual 需要 ASM 传播层：用 --propagator asm 或 hybrid")

    root = Path(args.data).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("数据集不完整，请先运行 generate.py")
    groups = manifest["groups"]
    if args.groups:
        wanted = set(float(x) for x in args.groups)
        groups = [g for g in groups if g["nf"] in wanted]
        if not groups:
            parser.error("--groups 指定的 NF 不在数据集中")
    if not groups:
        raise SystemExit("没有可训练的 NF 组")
    for split in ("train", "val", "test"):
        if not manifest["split_counts"][split]:
            raise SystemExit(f"{split} 划分为空，请增加 --samples")

    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    out = Path(args.out) if args.out else Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    out = out.resolve()
    if out.exists():
        raise SystemExit(f"输出目录已存在：{out}；请换一个新名字")
    out.mkdir(parents=True)
    with (out / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"arguments": vars(args), "device": str(device),
                   "data": str(root), "groups": [g["nf"] for g in groups]},
                  f, ensure_ascii=False, indent=2)

    print(f"设备：{device}；数据：{root}；输出：{out}", flush=True)
    rows = []
    for group in groups:
        rows.extend(run_group(root, group, args, device, out))
        write_csv(out / "summary.csv", rows)
    print(f"训练完成：{out}")


if __name__ == "__main__":
    main()
