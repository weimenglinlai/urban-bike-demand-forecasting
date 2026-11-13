"""
工具函数：日志、计时、画图、池化等通用功能
"""

import os
import sys
import time
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        log(f"{func.__name__} 完成，耗时 {elapsed:.2f}s")
        return result
    return wrapper


def save_fig(name: str, dpi: int = 150):
    path = os.path.join(config.FIGURE_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    log(f"图表已保存: {path}")
    return path


# ---------------------------------------------------------------------------
# 池化工具
# ---------------------------------------------------------------------------
# 当前架构（条件基线 + 残差学习，5×5 求和池化）只使用 pool_3d_sum：
# 求和池化保留区域总需求，避免平均池化把 16 个原始网格稀释为均值后
# 导致训练目标量级被压缩。以下四个函数是早期方案的遗留实现，主流程已无
# 调用点，保留作为方案对比与扩展备选（改回平均池化、或改用平滑上采样
# 呈现热力图时可直接启用）：
#   pool_5d   —— 5D 时空张量 (N,T,H,W,C) 平均池化
#   pool_3d   —— 3D (N,H,W) 平均池化
#   unpool    —— 双线性插值反池化（scipy.ndimage.zoom，热力图更连续）
#   unpool_5d —— 块状复制反池化（ConvLSTM 输出专用）


def pool_5d(arr, pool):
    """空间池化: (N, T, H, W, C) → (N, T, H//p, W//p, C)（遗留，未接入主流程）"""
    N, T, H, W, C = arr.shape
    if H % pool != 0 or W % pool != 0:
        raise ValueError(f"网格 {H}×{W} 不能被 pool={pool} 整除")
    return arr.reshape(N, T, H // pool, pool, W // pool, pool, C).mean(axis=(3, 5)).astype(np.float32)


def pool_3d(arr, pool):
    """空间池化: (N, H, W) → (N, H//p, W//p)，平均池化（遗留，未接入主流程）"""
    N, H, W = arr.shape
    if H % pool != 0 or W % pool != 0:
        raise ValueError(f"网格 {H}×{W} 不能被 pool={pool} 整除")
    return arr.reshape(N, H // pool, pool, W // pool, pool).mean(axis=(2, 4)).astype(np.float32)


def pool_3d_sum(arr, pool):
    """空间池化: (N, H, W) → (N, H//p, W//p)，求和池化（保留区域总需求）

    用于训练目标：平均池化会把 16 个原始网格稀释为均值，
    导致模型学到的量级被压缩、MSE 被大值压制。求和池化保留真实总量。
    """
    N, H, W = arr.shape
    if H % pool != 0 or W % pool != 0:
        raise ValueError(f"网格 {H}×{W} 不能被 pool={pool} 整除")
    return arr.reshape(N, H // pool, pool, W // pool, pool).sum(axis=(2, 4)).astype(np.float32)


def unpool(arr, pool, target_size):
    """
    双线性插值反池化: (N, Hp, Wp, ...) → (N, target_size, target_size, ...)
    将池化后的低分辨率预测平滑上采样回原始网格，保证热力图渐变连续。
    使用 scipy.ndimage.zoom bilinear (order=1)，比 np.repeat 块状复制自然得多。
    （遗留，未接入主流程；当前界面用 np.repeat 块状复制呈现。）
    """
    from scipy.ndimage import zoom
    if arr.ndim == 2:
        H, W = arr.shape
        z_h, z_w = target_size / H, target_size / W
        return zoom(arr.astype(np.float64), (z_h, z_w), order=1).astype(np.float32)
    if arr.ndim == 3:
        N, H, W = arr.shape
        z_h, z_w = target_size / H, target_size / W
        out = np.empty((N, target_size, target_size), dtype=np.float32)
        for i in range(N):
            out[i] = zoom(arr[i].astype(np.float64), (z_h, z_w), order=1).astype(np.float32)
        return out
    elif arr.ndim == 4:
        N, H, W, C = arr.shape
        z_h, z_w = target_size / H, target_size / W
        out = np.empty((N, target_size, target_size, C), dtype=np.float32)
        for i in range(N):
            for c in range(C):
                out[i, :, :, c] = zoom(arr[i, :, :, c].astype(np.float64), (z_h, z_w), order=1).astype(np.float32)
        return out
    else:
        raise ValueError(f"unpool 仅支持 2D/3D/4D 输入，收到 {arr.ndim}D")


def unpool_5d(pred_pooled, pool, gs):
    """
    5D 输入反池化: (N, Hp, Wp, 1) → (N, gs, gs)
    ConvLSTM 输出可用此函数。（遗留，未接入主流程）
    """
    N = pred_pooled.shape[0]
    flat = pred_pooled.reshape(N, -1, 1)
    broad = np.repeat(np.repeat(flat.reshape(N, -1), pool, axis=1).reshape(N, -1), pool, axis=1)
    return broad[:, :gs * gs].reshape(N, gs, gs).astype(np.float32)


# ---------------------------------------------------------------------------
# 缓存维护与出图工具
# ---------------------------------------------------------------------------
# 以下函数当前**没有调用点**，但不是死代码——它们是为「论文出图」与
# 「缓存自动化」预留的接口，删除会丢掉可复用的实现：
#   ensure_fresh_cache  —— DATA_VERSION 变更时自动清缓存的入口
#                          （现由 config.DATA_VERSION 记录 + 手动 --force 承担）
#   timer               —— 函数耗时装饰器
#   save_fig / plot_loss_curve / plot_scatter / plot_model_comparison
#                       —— 出图工具：训练损失曲线、预测-真实散点、
#                          四模型 RMSE/MAE/R² 对比柱状图
#                          （任务书要求的对比图可由这里产出，图形统一
#                            使用灰度配色，符合论文黑白印刷要求）


def ensure_fresh_cache():
    """检查 DATA_VERSION，版本变更时清空旧模型缓存"""
    ver_path = os.path.join(config.OUTPUT_DIR, ".data_version")
    current_ver = str(config.DATA_VERSION)
    stale = False
    if os.path.exists(ver_path):
        with open(ver_path, "r") as f:
            stale = f.read().strip() != current_ver
    else:
        stale = True
    if stale:
        log("数据版本变更，清空旧模型缓存")
        for f in os.listdir(config.OUTPUT_DIR):
            if f.endswith((".pth", ".pkl")):
                os.remove(os.path.join(config.OUTPUT_DIR, f))
        for f in os.listdir(config.FIGURE_DIR):
            if f.endswith(".png"):
                os.remove(os.path.join(config.FIGURE_DIR, f))
        with open(ver_path, "w") as f:
            f.write(current_ver)


def plot_loss_curve(train_losses, val_losses, title, save_name):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="训练 Loss", linewidth=1.5)
    if val_losses:
        plt.plot(val_losses, label="验证 Loss", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    return save_fig(save_name)


def plot_scatter(y_true, y_pred, title, save_name):
    y_t = np.array(y_true).flatten()
    y_p = np.array(y_pred).flatten()
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t, y_p = y_t[mask], y_p[mask]
    plt.figure(figsize=(7, 7))
    plt.scatter(y_t, y_p, alpha=0.3, s=2, edgecolors="none")
    vmin = min(y_t.min(), y_p.min())
    vmax = max(y_t.max(), y_p.max())
    plt.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1, label="理想线 y=x")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    return save_fig(save_name)


def plot_model_comparison(metrics_dict, save_name="model_comparison.png"):
    models = list(metrics_dict.keys())
    rmse_vals = [metrics_dict[m]["rmse"] for m in models]
    mae_vals = [metrics_dict[m]["mae"] for m in models]
    r2_vals = [metrics_dict[m]["r2"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#a0a0a0"] * len(models)

    for i, (vals, title, ylabel) in enumerate([
        (rmse_vals, "RMSE（越低越好）", "RMSE"),
        (mae_vals, "MAE（越低越好）", "MAE"),
        (r2_vals, "R2（越高越好）", "R2"),
    ]):
        c = colors.copy()
        if i < 2:
            c[np.argmin(vals)] = "#555555"
        else:
            c[np.argmax(vals)] = "#555555"
        axes[i].bar(models, vals, color=c, edgecolor="black", linewidth=0.5)
        axes[i].set_title(title)
        axes[i].set_ylabel(ylabel)
        for j, v in enumerate(vals):
            axes[i].text(j, v + max(vals) * 0.02, f"{v:.4f}", ha="center", fontsize=9)
        axes[i].grid(True, alpha=0.3, axis="y")

    return save_fig(save_name)
