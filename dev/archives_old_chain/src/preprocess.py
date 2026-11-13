"""
预处理与时空张量构造
输入：原始订单 DataFrame → 输出：训练/测试张量 + 归一化器 + 网格元数据
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_data
from src.utils import log, timer


@timer
def preprocess(force=False):
    """主预处理流程，返回处理后数据路径"""
    x_train_path = os.path.join(config.PROCESSED_DIR, "X_train.npy")
    if os.path.exists(x_train_path) and not force:
        log("预处理数据已存在，跳过。如需重新处理请设置 force=True")
        return config.PROCESSED_DIR

    log("=== 开始数据预处理 ===")

    df = load_data()
    log(f"原始数据: {len(df):,} 条")

    # 过滤骑行时长异常值（CitiBike 有 >24h 的脏数据）
    if "tripduration" in df.columns:
        dur_min = config.TRIP_DURATION_MIN
        dur_max = config.TRIP_DURATION_MAX
        before = len(df)
        df = df[(df["tripduration"] >= dur_min) & (df["tripduration"] <= dur_max)]
        log(f"骑行时长过滤 [{dur_min}s, {dur_max}s]: {before:,} → {len(df):,} 条")

    # 离群点剔除（1%-99% 分位数）
    for col in ["lng", "lat", "end_lng", "end_lat"]:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    log(f"离群点剔除后: {len(df):,} 条")

    # 网格划分（向量化）
    grid_meta = _build_grid(df)
    df["grid_x"], df["grid_y"] = _assign_grid(df, grid_meta)

    # 按小时+网格聚合
    df["hour"] = df["time"].dt.floor("h")
    counts = df.groupby(["hour", "grid_x", "grid_y"]).size().reset_index(name="count")
    log(f"网格聚合: {len(counts):,} 条记录，{counts['hour'].nunique()} 个时点")

    # 构建时空矩阵
    tensor = _build_tensor(counts, grid_meta)
    log(f"时空张量形状: {tensor.shape}")

    # 滑动窗口切分
    X, y = _sliding_window(tensor, config.WINDOW_SIZE)
    log(f"窗口切分: X={X.shape}, y={y.shape}")

    # 按时间顺序划分训练/测试集
    split = int(len(X) * config.TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    log(f"训练集: X={X_train.shape}, y={y_train.shape}")
    log(f"测试集: X={X_test.shape}, y={y_test.shape}")

    # 归一化
    X_train, X_test, scaler = _normalize(X_train, X_test)

    # 增加通道维度
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # 保存
    np.save(os.path.join(config.PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(config.PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(config.PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.PROCESSED_DIR, "y_test.npy"), y_test)

    with open(os.path.join(config.PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(config.PROCESSED_DIR, "grid_metadata.pkl"), "wb") as f:
        pickle.dump(grid_meta, f)

    log("=== 预处理完成，所有文件已保存到 data/processed/ ===")
    _print_summary(X_train, X_test, y_train, y_test, grid_meta)
    return config.PROCESSED_DIR


def _build_grid(df):
    """构建网格系统（向量化）"""
    gs = config.GRID_SIZE
    lng_edges = np.linspace(df["lng"].min(), df["lng"].max(), gs + 1)
    lat_edges = np.linspace(df["lat"].min(), df["lat"].max(), gs + 1)

    xi = np.clip(np.searchsorted(lng_edges, df["lng"].values, side="right") - 1, 0, gs - 1)
    yi = np.clip(np.searchsorted(lat_edges, df["lat"].values, side="right") - 1, 0, gs - 1)
    active_grids = set(zip(xi.tolist(), yi.tolist()))

    return {
        "grid_size": gs,
        "lng_edges": lng_edges,
        "lat_edges": lat_edges,
        "active_grids": active_grids,
        "lng_min": float(lng_edges[0]),
        "lng_max": float(lng_edges[-1]),
        "lat_min": float(lat_edges[0]),
        "lat_max": float(lat_edges[-1]),
    }


def _assign_grid(df, grid_meta):
    """为每条记录分配网格坐标（向量化）"""
    gs = grid_meta["grid_size"]
    xi = np.clip(np.searchsorted(grid_meta["lng_edges"], df["lng"].values, side="right") - 1, 0, gs - 1)
    yi = np.clip(np.searchsorted(grid_meta["lat_edges"], df["lat"].values, side="right") - 1, 0, gs - 1)
    return xi, yi


def _build_tensor(counts, grid_meta):
    """构建 (总小时数, grid, grid) 的时空矩阵"""
    hours = sorted(counts["hour"].unique())
    gs = grid_meta["grid_size"]
    tensor = np.zeros((len(hours), gs, gs), dtype=np.float32)

    hour_to_idx = {h: i for i, h in enumerate(hours)}
    for _, row in counts.iterrows():
        ti = hour_to_idx[row["hour"]]
        xi = int(row["grid_x"])
        yi = int(row["grid_y"])
        tensor[ti, xi, yi] = row["count"]

    return tensor


def _sliding_window(tensor, window):
    """滑动窗口切分"""
    X_list, y_list = [], []
    for i in range(len(tensor) - window):
        X_list.append(tensor[i : i + window])
        y_list.append(tensor[i + window])
    return np.array(X_list), np.array(y_list)


def _normalize(X_train, X_test):
    """训练集独立归一化，transform 测试集"""
    flat_train = X_train.reshape(X_train.shape[0], -1)
    flat_test = X_test.reshape(X_test.shape[0], -1)

    train_min = flat_train.min(axis=0)
    train_max = flat_train.max(axis=0)
    mask = (train_max == train_min)
    train_max[mask] = train_min[mask] + 1e-8
    train_min = train_min.reshape(1, -1)
    train_max = train_max.reshape(1, -1)

    flat_train = (flat_train - train_min) / (train_max - train_min)
    flat_test = (flat_test - train_min) / (train_max - train_min)
    flat_test = np.clip(flat_test, 0, 1)

    scaler = {"min": train_min, "max": train_max}
    return (
        flat_train.reshape(X_train.shape),
        flat_test.reshape(X_test.shape),
        scaler,
    )


def _print_summary(X_train, X_test, y_train, y_test, grid_meta):
    """打印预处理摘要"""
    print("\n" + "=" * 50)
    print("预处理结果摘要")
    print("=" * 50)
    print(f"  网格大小: {grid_meta['grid_size']}×{grid_meta['grid_size']}")
    print(f"  活跃网格数: {len(grid_meta['active_grids'])}/{grid_meta['grid_size']**2}")
    print(f"  地理范围: lng[{grid_meta['lng_min']:.3f}, {grid_meta['lng_max']:.3f}], lat[{grid_meta['lat_min']:.3f}, {grid_meta['lat_max']:.3f}]")
    print(f"  X_train: {X_train.shape}（{np.prod(X_train.shape):,} 个值）")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}（{np.prod(X_test.shape):,} 个值）")
    print(f"  y_test:  {y_test.shape}")
    print(f"  y_train 值范围: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    preprocess(force=False)
