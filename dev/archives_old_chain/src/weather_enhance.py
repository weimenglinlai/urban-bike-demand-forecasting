"""
天气特征增强：为已预处理的时空数据拼接天气通道
用法: python src/weather_enhance.py
"""

import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log
from src.weather import fetch_weather

CACHE_DIR = os.path.join(config.PROCESSED_DIR, "weather")


def enhance(force=False):
    """拉取天气并拼接为额外特征通道"""
    os.makedirs(CACHE_DIR, exist_ok=True)

    x_train_path = os.path.join(CACHE_DIR, "X_train.npy")
    if os.path.exists(x_train_path) and not force:
        log("天气增强数据已存在，跳过。")
        return

    X_train = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))

    log(f"X_train 当前 shape: {X_train.shape}")

    with open(os.path.join(config.PROCESSED_DIR, "grid_metadata.pkl"), "rb") as f:
        grid_meta = pickle.load(f)

    from src.data_loader import load_data
    df = load_data()
    t_min = df["time"].min()
    t_max = df["time"].max()
    log(f"数据时间范围: {t_min} ~ {t_max}")

    center_lat = (grid_meta["lat_min"] + grid_meta["lat_max"]) / 2
    center_lng = (grid_meta["lng_min"] + grid_meta["lng_max"]) / 2

    weather_df = fetch_weather(
        center_lat, center_lng,
        t_min.strftime("%Y-%m-%d"),
        t_max.strftime("%Y-%m-%d"),
    )

    weather_df = weather_df.drop_duplicates(subset="time").set_index("time").sort_index()
    hour_range = pd.date_range(t_min.floor("h"), t_max.ceil("h"), freq="h")
    weather_df = weather_df.reindex(hour_range).ffill().bfill()

    w_features = weather_df[["temperature", "precipitation", "wind_speed", "cloud_cover"]].values.astype(np.float32)

    w_mean = w_features.mean(axis=0)
    w_std = w_features.std(axis=0) + 1e-8
    w_features = (w_features - w_mean) / w_std

    # 保存归一化参数（滚动预测未来天气时必需）
    np.save(os.path.join(CACHE_DIR, "weather_mean.npy"), w_mean)
    np.save(os.path.join(CACHE_DIR, "weather_std.npy"), w_std)

    saved_weather = {
        "features": w_features,
        "mean": w_mean,
        "std": w_std,
        "time_range": weather_df.index,
    }
    np.save(os.path.join(CACHE_DIR, "weather_features.npy"), w_features)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    window = X_train.shape[1]
    gs = config.GRID_SIZE
    n_channels = 1 + 4

    # 对于每个样本窗口的每个时间步，精确填入该小时的天气特征
    def _concat(X_arr, n_samp, offset):
        new_x = np.zeros((n_samp, window, gs, gs, n_channels), dtype=np.float32)
        for i in range(n_samp):
            new_x[i, :, :, :, 0] = X_arr[i, :, :, :, 0]
            for t in range(window):
                widx = offset + i + t
                if 0 <= widx < len(w_features):
                    new_x[i, t, :, :, 1:5] = w_features[widx, :]
                else:
                    new_x[i, t, :, :, 1:5] = 0.0
        return new_x

    # 训练集的每个样本对应前 window 个时间步的天气
    X_train_w = _concat(X_train, n_train, offset=0)
    # 测试集紧接训练集之后
    X_test_w = _concat(X_test, n_test, offset=n_train)

    np.save(os.path.join(CACHE_DIR, "X_train.npy"), X_train_w)
    np.save(os.path.join(CACHE_DIR, "X_test.npy"), X_test_w)
    np.save(os.path.join(CACHE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(CACHE_DIR, "y_test.npy"), y_test)

    log(f"天气增强完成: {X_train_w.shape}, 通道={n_channels}（逐小时精确填充）")
    return CACHE_DIR


def load_enhanced_data():
    """加载天气增强后的数据，fallback 到原始数据"""
    x_train_path = os.path.join(CACHE_DIR, "X_train.npy")
    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path)
        X_test = np.load(os.path.join(CACHE_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(CACHE_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(CACHE_DIR, "y_test.npy"))
        log(f"加载天气增强数据: X_train={X_train.shape}")
        return X_train, X_test, y_train, y_test, X_train.shape[-1]

    X_train = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test, X_train.shape[-1]


if __name__ == "__main__":
    enhance(force=True)
