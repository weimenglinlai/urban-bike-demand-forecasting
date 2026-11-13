"""
周期特征预测模型（Period Model）
================================
核心思路：不依赖"滑动窗口 + 递归滚动"，而是直接学习需求的周期性规律：
  - 一天内的潮汐（小时 0-23：早高峰/晚高峰/夜间低谷）
  - 一周内的模式（周一~周五通勤 vs 周末休闲）
  - 节假日效应（中秋/国庆等）
  - 天气影响（温度/降水/风速/云量）
  - 事件扰动（活动/事故，预测时叠加）

因此可以预测**任意未来日期**（只需知道那天是星期几 + 天气情景），
不受"训练数据之后十几天"的限制；并标注模型有效期，提醒用最新数据重训。

用法:
  python src/period_model.py            # 训练并保存
  python src/period_model.py --eval     # 训练 + 回测评估
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_data
from src.utils import log, pool_3d, unpool

MODEL_PATH = os.path.join(config.OUTPUT_DIR, "period_model.pkl")

# ============ 节假日表（2018 中国法定节假日） ============
HOLIDAYS_2018 = {
    "2018-09-24": "中秋节",
    "2018-10-01": "国庆节",
    "2018-10-02": "国庆节",
    "2018-10-03": "国庆节",
    "2018-10-04": "国庆节",
    "2018-10-05": "国庆节",
    "2018-10-06": "国庆节",
    "2018-10-07": "国庆节",
}

# ============ 天气情景 → 归一化特征向量 ============
# 校准到训练分布内（weather_mean=[21.07, 0.36, 12.28, 72.84],
#                     weather_std =[3.80, 1.03, 5.93, 35.89]）
WEATHER_SCENARIOS = {
    "晴天": np.array([26.0, 0.0, 10.0, 30.0]),      # 温度26°C 无雨 微风 少云
    "阴天": np.array([24.0, 0.5, 14.0, 80.0]),      # 温度24°C 微量 中风 多云
    "小雨": np.array([22.0, 3.0, 16.0, 90.0]),      # 温度22°C 小雨 中风 阴
    "大雨": np.array([20.0, 8.0, 22.0, 95.0]),      # 温度20°C 大雨 强风 全阴
    "高温": np.array([32.0, 0.0, 12.0, 40.0]),      # 高温 无雨 微风
}


def scenario_to_weather_vector(scenario: str):
    """天气情景 → z-score 特征向量（温度/降水/风速/云量）"""
    w_mean = np.array([21.07, 0.36, 12.28, 72.84])
    w_std = np.array([3.80, 1.03, 5.93, 35.89])
    raw = WEATHER_SCENARIOS.get(scenario, WEATHER_SCENARIOS["晴天"]).astype(np.float64)
    return (raw - w_mean) / w_std


# ============ 数据准备 ============

def build_hourly_demand():
    """
    从原始订单构建逐小时需求张量 (n_hours, gs, gs) + 时间戳数组。
    复用与 preprocess 相同的网格划分与过滤逻辑。
    """
    df = load_data()

    if "tripduration" in df.columns:
        dur_min = config.TRIP_DURATION_MIN
        dur_max = config.TRIP_DURATION_MAX
        df = df[(df["tripduration"] >= dur_min) & (df["tripduration"] <= dur_max)]

    for col in ["lng", "lat", "end_lng", "end_lat"]:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    # 网格划分（与 preprocess 相同）
    gs = config.GRID_SIZE
    lng_edges = np.linspace(df["lng"].min(), df["lng"].max(), gs + 1)
    lat_edges = np.linspace(df["lat"].min(), df["lat"].max(), gs + 1)
    xi = np.clip(np.searchsorted(lng_edges, df["lng"].values, side="right") - 1, 0, gs - 1)
    yi = np.clip(np.searchsorted(lat_edges, df["lat"].values, side="right") - 1, 0, gs - 1)
    df["grid_x"], df["grid_y"] = xi, yi

    df["hour"] = df["time"].dt.floor("h")
    counts = df.groupby(["hour", "grid_x", "grid_y"]).size().reset_index(name="count")

    hours = sorted(counts["hour"].unique())
    tensor = np.zeros((len(hours), gs, gs), dtype=np.float32)
    hour_to_idx = {h: i for i, h in enumerate(hours)}
    for _, row in counts.iterrows():
        tensor[hour_to_idx[row["hour"]], int(row["grid_x"]), int(row["grid_y"])] = row["count"]

    return tensor, np.array(hours)


def _load_weather_aligned(hours):
    """加载天气特征并对齐到逐小时时间戳 (n_hours, 4)"""
    w_path = os.path.join(config.PROCESSED_DIR, "weather", "weather_features.npy")
    if os.path.exists(w_path):
        w_features = np.load(w_path)  # (721, 4) 9/1 00:00 ~ 10/1 00:00
        df = load_data()
        t0 = np.datetime64(df["time"].min().floor("h"))
        hours64 = hours.astype("datetime64[h]")
        idx = ((hours64 - t0) / np.timedelta64(1, "h")).astype(int)
        idx = np.clip(idx, 0, len(w_features) - 1)
        return w_features[idx].astype(np.float32)
    # fallback：无天气数据时用 0（等价于平均天气）
    log("未找到天气特征文件，天气通道置 0")
    return np.zeros((len(hours), 4), dtype=np.float32)


def build_features(hours, weather_arr):
    """构造周期 + 天气特征矩阵 (n_hours, n_features)"""
    n = len(hours)
    ts = pd.to_datetime(hours)
    hour = ts.hour.values.astype(np.float64)
    dow = ts.dayofweek.values.astype(np.float64)  # 0=周一
    is_weekend = ((dow >= 5)).astype(np.float64)

    is_holiday = np.array([
        1.0 if d.strftime("%Y-%m-%d") in HOLIDAYS_2018 else 0.0
        for d in ts
    ])

    feats = np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        is_weekend, is_holiday,
        weather_arr,  # 4 维
    ]).astype(np.float32)
    return feats


FEATURE_NAMES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                 "is_weekend", "is_holiday",
                 "temp", "precip", "wind", "cloud"]


# ============ 训练 ============

def train(force=False):
    """训练周期模型：25 个池化网格各一个 GBDT"""
    if os.path.exists(MODEL_PATH) and not force:
        log(f"周期模型已存在: {MODEL_PATH}，跳过（--force 强制重训）")
        return MODEL_PATH

    from sklearn.ensemble import HistGradientBoostingRegressor

    log("=== 训练周期特征模型 ===")
    tensor, hours = build_hourly_demand()
    log(f"逐小时需求: {tensor.shape}（{len(hours)} 小时）")

    pool = config.XGB_POOL
    pooled = pool_3d(tensor, pool)  # (n_hours, 5, 5)
    log(f"池化到 {pooled.shape[1]}×{pooled.shape[2]} = {pooled.shape[1]*pooled.shape[2]} 网格")

    weather_arr = _load_weather_aligned(hours)
    feats = build_features(hours, weather_arr)
    log(f"特征矩阵: {feats.shape}，特征: {FEATURE_NAMES}")

    # 时间顺序划分：前 85% 训练，后 15% 测试
    split = int(len(feats) * 0.85)
    X_train, X_test = feats[:split], feats[split:]
    y_train_pool = pooled[:split]
    y_test_pool = pooled[split:]
    train_hours, test_hours = hours[:split], hours[split:]
    log(f"训练: {X_train.shape[0]} 小时 | 测试: {X_test.shape[0]} 小时")

    # 每个池化网格一个 GBDT
    n_cells = pooled.shape[1] * pooled.shape[2]
    models = []
    t0 = time.time()
    for c in range(n_cells):
        gx, gy = divmod(c, pooled.shape[2])
        m = HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.08,
            random_state=config.RANDOM_SEED,
        )
        m.fit(X_train, y_train_pool[:, gx, gy])
        models.append(m)
    elapsed = time.time() - t0
    log(f"训练完成: {n_cells} 个 GBDT，耗时 {elapsed:.1f}s")

    # 测试集评估
    pred_pool = np.stack([m.predict(X_test) for m in models], axis=-1)
    pred_pool = pred_pool.reshape(-1, pooled.shape[1], pooled.shape[2])
    y_test_flat = y_test_pool.reshape(-1)
    pred_flat = pred_pool.reshape(-1)
    rmse = float(np.sqrt(np.mean((y_test_flat - pred_flat) ** 2)))
    mae = float(np.mean(np.abs(y_test_flat - pred_flat)))
    ss_res = np.sum((y_test_flat - pred_flat) ** 2)
    ss_tot = np.sum((y_test_flat - np.mean(y_test_flat)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    log(f"测试集（池化网格级）: RMSE={rmse:.4f} MAE={mae:.4f} R²={r2:.4f}")

    # 逐小时总需求对比
    test_daily = y_test_pool.sum(axis=(1, 2))
    pred_daily = pred_pool.sum(axis=(1, 2))
    log(f"测试集逐小时总需求 真实均值 {test_daily.mean():.1f} vs 预测均值 {pred_daily.mean():.1f}")

    # 模型有效期信息
    data_min, data_max = pd.to_datetime(hours).min(), pd.to_datetime(hours).max()
    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_range": [str(data_min.date()), str(data_max.date())],
        "n_hours": len(hours),
        "grid_size": config.GRID_SIZE,
        "pool": pool,
        "feature_names": FEATURE_NAMES,
        "test_metrics": {"rmse": rmse, "mae": mae, "r2": r2},
        "suggest_retrain_days": 30,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"models": models, "meta": meta, "pool": pool}, f)
    log(f"模型已保存: {MODEL_PATH}")
    return MODEL_PATH


# ============ 预测 ============

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_hour(target_date, hour, weather_scenario="晴天", event=None, model=None):
    """
    预测目标日期某一小时的需求 (gs, gs)。
    target_date: 'YYYY-MM-DD'（任意未来/过去日期）
    hour: 0-23
    """
    if model is None:
        model = load_model()
    if model is None:
        return None

    ts = pd.Timestamp(f"{target_date} {hour:02d}:00:00")
    d = ts.strftime("%Y-%m-%d")
    is_holiday = 1.0 if d in HOLIDAYS_2018 else 0.0
    dow = ts.dayofweek

    wv = scenario_to_weather_vector(weather_scenario)
    feats = np.array([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        1.0 if dow >= 5 else 0.0, is_holiday,
        *wv,
    ], dtype=np.float32).reshape(1, -1)

    pool = model["pool"]
    gs = model["meta"]["grid_size"]
    pg = gs // pool
    pred_pool = np.array([m.predict(feats)[0] for m in model["models"]]).reshape(pg, pg)
    pred = unpool(pred_pool, pool, gs)
    pred = np.clip(pred, 0, None)

    # 事件扰动
    if event is not None and event.get("type") in ("activity", "accident"):
        ex, ey = int(event["x"]), int(event["y"])
        radius = int(event.get("radius", 2))
        strength = float(event.get("strength", 0.3))
        mask = np.zeros((gs, gs), dtype=bool)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = ex + dx, ey + dy
                if 0 <= nx < gs and 0 <= ny < gs:
                    mask[nx, ny] = True
        if event["type"] == "activity":
            pred[mask] = pred[mask] * (1.0 + strength)
        else:
            pred[mask] = pred[mask] * max(0.0, 1.0 - strength * 0.8)
        pred = np.clip(pred, 0, None)

    return pred


def predict_day(target_date, weather_scenario="晴天", event=None, model=None):
    """预测目标日期全天 24 小时 → (24, gs, gs)"""
    if model is None:
        model = load_model()
    if model is None:
        return None
    out = np.stack([
        predict_hour(target_date, h, weather_scenario, event, model)
        for h in range(24)
    ])
    return out


def get_model_info():
    """模型有效期等元信息（前端展示用）"""
    model = load_model()
    if model is None:
        return None
    return model["meta"]


# ============ 主入口 ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重训")
    parser.add_argument("--eval", action="store_true", help="训练后回测评估")
    args = parser.parse_args()

    train(force=args.force)

    if args.eval or True:
        # 演示：预测 10/01（周二 国庆）、10/06（周日 国庆）、9/24（周一 中秋）
        model = load_model()
        meta = model["meta"]
        print("\n" + "=" * 60)
        print("模型信息")
        print("=" * 60)
        print(f"  训练时间: {meta['trained_at']}")
        print(f"  数据范围: {meta['data_range'][0]} ~ {meta['data_range'][1]}")
        print(f"  测试指标: RMSE={meta['test_metrics']['rmse']:.3f} MAE={meta['test_metrics']['mae']:.3f} R²={meta['test_metrics']['r2']:.3f}")
        print(f"  建议重训周期: {meta['suggest_retrain_days']} 天")

        for date, w in [("2018-10-01", "晴天"), ("2018-10-01", "大雨"), ("2018-10-06", "晴天"), ("2018-09-24", "晴天")]:
            pred = predict_hour(date, 8, w)
            if pred is not None:
                print(f"\n  {date} 08:00 {w}: 总需求 {pred.sum():.0f} 辆 | 最大网格 {pred.max():.1f}")

        # 未来任意日期（验证不依赖训练数据范围）
        for date in ["2018-11-15", "2019-01-01", "2026-08-13"]:
            pred = predict_hour(date, 8, "晴天")
            if pred is not None:
                print(f"  {date} 08:00 晴天: 总需求 {pred.sum():.0f} 辆（星期 {pd.Timestamp(date).day_name()}）")
