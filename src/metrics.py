"""
统一评估指标模块（Metrics）
============================
所有模型共用同一套指标与同一份掩码，保证「同口径对照」。

指标：
  RMSE / MAE / R² / MAPE（相对误差，仅在真实值 ≥ MAPE_MIN_TRUE 的样本上计算，
  并显式报告参与计算的样本占比）/ WAPE（加权绝对百分比误差，对零值稳健）

细分维度：
  时段（高峰 / 平峰）、日期类型（工作日 / 周末 / 法定假日 / 调休上班日）、
  天气（有雨 / 无雨）
"""

import numpy as np

from src.date_features import DAY_TYPE_NAMES, RAIN_LABELS

# MAPE 的分母保护阈值：池化区块小时需求低于该值时不计入 MAPE
MAPE_MIN_TRUE = 1.0
# 「活跃区块」阈值：需求达到该量级的样本上另算 MAPE，
# 因为池化后存在大量接近 0 的冷门区块，会主导 MAPE 的绝对值
ACTIVE_MIN_TRUE = 100.0

# 高峰时段（通勤双峰）
PEAK_HOURS = (7, 8, 9, 17, 18, 19)


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def core_metrics(y_true, y_pred) -> dict:
    """RMSE / MAE / R² / MAPE / WAPE（输入为一维数组）。"""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = _r2(yt, yp)
    m = yt >= MAPE_MIN_TRUE
    denom = np.clip(yt[m], MAPE_MIN_TRUE, None)
    mape = float(np.mean(np.abs(err[m]) / denom) * 100.0) if m.any() else float("nan")
    ma = yt >= ACTIVE_MIN_TRUE
    mape_active = float(np.mean(np.abs(err[ma]) / yt[ma]) * 100.0) if ma.any() else float("nan")
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(yt)), 1e-9) * 100.0)
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "mape": mape, "mape_coverage": float(m.mean()),
        "mape_active": mape_active, "mape_active_coverage": float(ma.mean()),
        "wape": wape, "n": int(yt.size),
    }


def segmented_metrics(y_true, y_pred, hours, dt_arr, rb_arr) -> dict:
    """按 时段 / 日期类型 / 天气 三个维度细分。

    y_true / y_pred : (n, pg, pg)
    hours           : 长度 n 的时间戳（或 DatetimeIndex）
    dt_arr          : (n,) 日期类型索引
    rb_arr          : (n,) 降雨档索引
    """
    import pandas as pd

    yt = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)
    ts = pd.to_datetime(hours)
    hod = ts.hour.values
    dt_arr = np.asarray(dt_arr)
    rb_arr = np.asarray(rb_arr)

    out = {}

    def _add(name, mask):
        if mask.sum() == 0:
            return
        out[name] = core_metrics(yt[mask].ravel(), yp[mask].ravel())
        out[name]["n_hours"] = int(mask.sum())

    peak = np.isin(hod, PEAK_HOURS)
    _add("高峰时段", peak)
    _add("平峰时段", ~peak)

    for k, name in enumerate(DAY_TYPE_NAMES):
        _add(name, dt_arr == k)

    _add("有雨", rb_arr >= 1)
    _add("无雨", rb_arr == 0)
    for k, name in enumerate(RAIN_LABELS):
        if k == 0:
            continue
        _add(f"降雨-{name}", rb_arr == k)
    return out


def per_cell_r2(y_true, y_pred) -> np.ndarray:
    """逐区块的时间维 R²（剔除空间方差的影响，用于补充诊断）。"""
    yt = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)
    out = np.full(yt.shape[1], np.nan)
    for c in range(yt.shape[1]):
        v = float(np.var(yt[:, c]))
        if v <= 1e-9:
            continue
        out[c] = 1.0 - float(np.mean((yt[:, c] - yp[:, c]) ** 2)) / v
    return out


def format_metrics(m: dict) -> str:
    mape = m.get("mape", float("nan"))
    ma = m.get("mape_active", float("nan"))
    return (f"RMSE={m['rmse']:.2f} MAE={m['mae']:.2f} R²={m['r2']:.4f} "
            f"MAPE={mape:.1f}%(覆盖{m.get('mape_coverage', 0)*100:.0f}%) "
            f"MAPE活跃={ma:.1f}% WAPE={m['wape']:.1f}%")
