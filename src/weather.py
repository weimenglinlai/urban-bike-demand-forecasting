"""
天气特征提取模块
从 Open-Meteo API 拉取历史天气数据（免费，无需 API Key）
输出：每小时温度、降水量、风速等特征，归一化后作为额外通道
"""

import os, sys, json, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log


def fetch_weather(lat, lng, start_date, end_date):
    """
    从 Open-Meteo Archive API 拉取历史天气。
    返回 DataFrame，列: temperature_2m, precipitation, wind_speed_10m, cloud_cover
    """
    import openmeteo_requests

    log(f"获取天气: {start_date} ~ {end_date}, ({lat:.2f}, {lng:.2f})")

    client = openmeteo_requests.Client()

    try:
        responses = client.weather_api(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lng,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"],
                "timezone": "auto",
            },
        )
        resp = responses[0]
        hourly = resp.Hourly()
        if not hourly:
            raise ValueError("无天气数据返回")

        # 新版 openmeteo_requests: Time() 返回起始时间戳(int)，配合 Interval() 生成完整序列
        t0 = hourly.Time()
        interval = hourly.Interval() if hourly.Interval() else 3600
        temp = hourly.Variables(0).ValuesAsNumpy()
        precip = hourly.Variables(1).ValuesAsNumpy()
        wind = hourly.Variables(2).ValuesAsNumpy()
        cloud = hourly.Variables(3).ValuesAsNumpy()
        n = len(temp)
        # Open-Meteo Time() 返回 UTC 时间戳；timezone=auto 数据是本地时区序列，
        # 需先将 UTC 时间戳转换为本地时区（深圳 UTC+8），否则时间标签整体偏早 8 小时
        t = (pd.to_datetime(t0, unit="s", utc=True)
             .tz_convert("Asia/Shanghai").tz_localize(None)
             + pd.to_timedelta(np.arange(n) * interval, unit="s"))

        df = pd.DataFrame({
            "time": t,
            "temperature": temp,
            "precipitation": precip,
            "wind_speed": wind,
            "cloud_cover": cloud,
        })
        log(f"获取成功: {len(df)} 条天气记录")
        return df

    except Exception as e:
        log(f"API 失败: {e}，使用默认值")
        return _generate_default_weather(start_date, end_date)


def _generate_default_weather(start_date, end_date):
    """生成默认天气（秋季温和）"""
    hours = pd.date_range(start_date, end_date, freq="h")
    n = len(hours)
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "time": hours,
        "temperature": 20 + rng.normal(0, 5, n),
        "precipitation": np.maximum(rng.exponential(0.5, n) - 0.3, 0),
        "wind_speed": np.maximum(rng.exponential(3, n), 0.5),
        "cloud_cover": np.clip(rng.normal(40, 30, n), 0, 100),
    })


def merge_weather_to_tensor(X, y, df_orders, grid_meta, force=False):
    """
    将天气特征作为额外通道拼接到 X 张量。
    X: (n, window, h, w, channels) — 当前 channels=1
    df_orders: 原始订单数据（含 time 列，用于确定时间范围）
    返回: 新的 X (n, window, h, w, channels+4)

    如果 force=False 且天气特征已存在，直接加载。
    """
    cache_path = os.path.join(config.PROCESSED_DIR, "X_weather_extra.npy")
    if os.path.exists(cache_path) and not force:
        log("天气特征已缓存，加载中…")
        return np.load(cache_path)

    log("=== 构建天气特征通道 ===")

    with open(os.path.join(config.PROCESSED_DIR, "scaler.pkl"), "rb") as f:
        import pickle
        scaler = pickle.load(f)

    n, window, gs, gs_y, ch = X.shape

    center_lat = (grid_meta["lat_min"] + grid_meta["lat_max"]) / 2
    center_lng = (grid_meta["lng_min"] + grid_meta["lng_max"]) / 2

    t_min = df_orders["time"].min().strftime("%Y-%m-%d")
    t_max = df_orders["time"].max().strftime("%Y-%m-%d")

    weather_df = fetch_weather(center_lat, center_lng, t_min, t_max)

    hour_series = pd.date_range(t_min, t_max, freq="h")
    weather_df = weather_df.set_index("time").reindex(hour_series).fillna(0).reset_index()
    weather_df.rename(columns={"index": "time"}, inplace=True)

    weather_features = weather_df[["temperature", "precipitation", "wind_speed", "cloud_cover"]].values
    weather_features = weather_features[:n + window]

    weather_features = (weather_features - weather_features.mean(axis=0)) / (weather_features.std(axis=0) + 1e-8)

    X_new = np.zeros((n, window, gs, gs_y, ch + 4), dtype=np.float32)
    X_new[..., 0] = X[..., 0]

    for i in range(n):
        for t in range(window):
            idx = i + t
            if idx < len(weather_features):
                for c in range(4):
                    X_new[i, t, :, :, c + 1] = weather_features[idx, c]

    np.save(cache_path, X_new)
    log(f"天气特征拼接完成: {X_new.shape}，已保存")
    return X_new


if __name__ == "__main__":
    import pickle
    df = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "orders_weather.parquet")) if os.path.exists(
        os.path.join(config.PROCESSED_DIR, "orders_weather.parquet")
    ) else None

    if df is None:
        from src.data_loader import load_data
        df = load_data()

    with open(os.path.join(config.PROCESSED_DIR, "grid_metadata.pkl"), "rb") as f:
        grid_meta = pickle.load(f)

    X = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))
    y = np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy"))
    merge_weather_to_tensor(X, y, df, grid_meta, force=True)
