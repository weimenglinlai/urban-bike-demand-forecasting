"""
周期特征训练器（Period Trainer）
=================================
统一框架：**条件基线 + 残差学习**

    y(t) = B(t) + r(t)

  - B(t)：条件基线（src/baseline.py），由「小时 × 日期类型 × 降雨档」的
    训练集条件均值给出，无学习参数、可复现、对未来任意日期可用。
  - r(t)：残差，由模型学习。

四个模型（全部共享同一数据集、同一时间切分、同一目标分解、同一评估协议）：

| 键 | 名称 | 机理类别 | 说明 |
|----|------|----------|------|
| `gbt`      | 梯度提升树 | 传统机器学习 | 逐区块一个 HistGBR，输入 z(t) |
| `mlp`      | 多层感知机 | 消融对照（无序列机制） | 输入 z(t)，用于隔离「序列机制」的贡献 |
| `lstm`     | 残差序列 LSTM | 时序深度学习 | 循环 24h **残差序列** r(t−24..t−1)，外生上下文门控融合 |
| `convlstm` | 残差张量 ConvLSTM + FiLM | 时空深度学习 | 在 5×5 **残差图序列**上卷积，外生条件经 FiLM 逐通道调制 |

为什么这样改（对原设计的修正）
--------------------------------
原实现把 10 维「周期 + 气象」特征直接排成 24 步序列喂给 LSTM/ConvLSTM：
  * 每步输入完全相同的信息（小时已由 sin/cos 编码），序列中**无可用信息**，
    LSTM 的门控退化为普通全连接；
  * ConvLSTM 更把同一个特征向量广播到 5×5 全部网格，**空间卷积无对象**；
  * 预测时气象被铺成 24 小时常数（训练时是逐小时变化），
    造成训练—推理分布不一致，预测系统性偏移 41%~69%。

改后：
  * 序列循环的是**残差**（幅度小、近似零均值、平稳），节假日与天气等强冲击
    已由 B(t) 吸收，因此「循环会因节假日/天气而失准」的问题被结构性规避；
  * ConvLSTM 的输入是真正的 5×5 残差图 + 静态空间先验图，空间卷积有对象；
  * 外生条件通过 FiLM 逐通道调制卷积门控，而不是广播常量；
  * 气象情景按**典型日廓线逐小时展开**，训练与推理由此同分布。

两种评估口径
------------
  * **冷启动（cold）**：无历史残差观测——纯未来预测场景（App 默认口径）。
    此时四个模型看到的信息完全相同，构成严格的同口径消融对照。
  * **滚动（rolling）**：提供目标时刻前 24h 的真实残差——短期滚动预测场景。
    用于量化「序列机制」在信息可得时的增益。

用法:
  python src/period_trainer.py                  # 训练全部模型
  python src/period_trainer.py --force          # 强制重训
  python src/period_trainer.py --models gbt,mlp # 只训指定模型
"""

import argparse
import os
import pickle
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log, pool_3d_sum
from src.date_features import (
    FEATURE_NAMES, N_FEATURES,
    day_type_array, rain_bin, exogenous_features,
    build_scenario_profiles, save_scenario_profiles, scenario_hourly_weather,
    WEATHER_SCENARIOS,
)
from src.baseline import ConditionalBaseline
from src.metrics import core_metrics, segmented_metrics, per_cell_r2, format_metrics

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(config.OUTPUT_DIR, "period_models.pkl")

_MODULE_NAME = "src.period_trainer"


# ============================================================================
# 一、数据准备
# ============================================================================

def build_hourly_demand(force=False):
    """从原始订单构建逐小时需求张量 (n_hours, gs, gs) + 时间戳数组 + 网格元数据。

    流式 chunk 聚合：逐 chunk 读取 CSV → 空间分箱 → 逐小时聚合 → 累加。
    避免将数亿条数据全部加载到内存。
    """
    import pickle as _pk
    cache_tensor = os.path.join(config.PROCESSED_DIR, 'hourly_tensor.npy')
    cache_hours = os.path.join(config.PROCESSED_DIR, 'hourly_hours.npy')
    cache_grid = os.path.join(config.PROCESSED_DIR, 'grid_metadata.pkl')

    if not force and os.path.exists(cache_tensor) and os.path.exists(cache_hours) and os.path.exists(cache_grid):
        tensor = np.load(cache_tensor)
        hours = np.load(cache_hours, allow_pickle=True)
        with open(cache_grid, 'rb') as f:
            grid_meta = _pk.load(f)
        log(f'加载缓存逐小时需求: {tensor.shape} ({len(hours)} 小时)')
        return tensor, np.array(hours), grid_meta

    from src.data_loader import _detect_encoding, FIELD_MAP

    def _list_csv(path):
        if not path or not os.path.isdir(path):
            return []
        return sorted(f for f in os.listdir(path) if f.lower().endswith('.csv'))

    day_files = _list_csv(getattr(config, 'RAW_DAY_DIR', None))
    raw_files = _list_csv(config.RAW_DIR)
    if not day_files and not raw_files:
        raise FileNotFoundError(f'无 CSV 数据源（检查 {getattr(config, "RAW_DAY_DIR", None)} 与 {config.RAW_DIR}）')

    src_files = [os.path.join(config.RAW_DAY_DIR, f) for f in day_files] if day_files else \
                [os.path.join(config.RAW_DIR, f) for f in raw_files]
    log(f'数据源: {len(src_files)} 个 CSV 文件')
    if len(src_files) > 1:
        log(f'  首个: {os.path.basename(src_files[0])} | 末尾: {os.path.basename(src_files[-1])}')

    enc = _detect_encoding(src_files[0])
    header = pd.read_csv(src_files[0], nrows=0, encoding=enc)
    cols = {c.upper(): c for c in header.columns}
    field_map_upper = {k.upper(): v for k, v in FIELD_MAP.items()}

    def _pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        for n in names:
            un = n.upper()
            if un in field_map_upper:
                return cols.get(field_map_upper[un])
        for c in header.columns:
            if c.upper() in field_map_upper:
                return c
        return None

    time_col = _pick('START_TIME', 'start_time')
    lng_col = _pick('START_LNG', 'start_lng', 'lng')
    lat_col = _pick('START_LAT', 'start_lat', 'lat')
    if time_col is None or lng_col is None or lat_col is None:
        raise KeyError(f'CSV 缺少必需列, header={list(header.columns)}')

    log('第一遍扫描: 采样估算坐标范围...')
    lng_samples, lat_samples = [], []
    t_min_val = t_max_val = None
    total_rows = 0
    sample_frac = config.COORD_SAMPLE_FRAC
    rng = np.random.default_rng(42)
    for path in src_files:
        reader = pd.read_csv(path, encoding=enc, skiprows=1,
                             names=list(header.columns), chunksize=config.ORDER_CHUNK_SIZE)
        for chunk in reader:
            total_rows += len(chunk)
            keep = rng.random(len(chunk)) < sample_frac
            if keep.any():
                lng_samples.append(chunk.loc[keep, lng_col].values.astype(np.float32))
                lat_samples.append(chunk.loc[keep, lat_col].values.astype(np.float32))
            t_vals = pd.to_datetime(chunk[time_col], errors='coerce')
            cmin, cmax = t_vals.min(), t_vals.max()
            if cmin is not pd.NaT and (t_min_val is None or cmin < t_min_val):
                t_min_val = cmin
            if cmax is not pd.NaT and (t_max_val is None or cmax > t_max_val):
                t_max_val = cmax
            if total_rows % 10000000 == 0:
                log(f'  第一遍已扫描 {total_rows/1e6:.0f}M 行...')
    lng_arr = np.concatenate(lng_samples)
    lat_arr = np.concatenate(lat_samples)
    del lng_samples, lat_samples
    log(f'  采样样本: {len(lng_arr):,} 个坐标点 ({100*len(lng_arr)/max(total_rows,1):.2f}%)')
    lng_lo, lng_hi = np.nanpercentile(lng_arr, 1), np.nanpercentile(lng_arr, 99)
    lat_lo, lat_hi = np.nanpercentile(lat_arr, 1), np.nanpercentile(lat_arr, 99)
    del lng_arr, lat_arr
    log(f'第一遍完成: 共 {total_rows} 行')
    log(f'坐标范围: lng[{lng_lo:.4f}, {lng_hi:.4f}] lat[{lat_lo:.4f}, {lat_hi:.4f}]')
    log(f'时间范围: {t_min_val} ~ {t_max_val}')

    gs = config.GRID_SIZE
    lng_edges = np.linspace(lng_lo, lng_hi, gs + 1)
    lat_edges = np.linspace(lat_lo, lat_hi, gs + 1)

    log('第二遍扫描: 逐 chunk 聚合逐小时网格需求...')
    from collections import defaultdict
    hour_grid = defaultdict(lambda: np.zeros((gs, gs), dtype=np.float32))
    chunk_size = config.ORDER_CHUNK_SIZE
    n_processed = 0
    for path in src_files:
        reader = pd.read_csv(path, encoding=enc, chunksize=chunk_size,
                             dtype={lng_col: np.float32, lat_col: np.float32})
        for chunk in reader:
            t = pd.to_datetime(chunk[time_col], errors='coerce')
            lng = chunk[lng_col].values
            lat = chunk[lat_col].values
            valid = (~t.isna()) & (~np.isnan(lng)) & (~np.isnan(lat)) & \
                    (lng >= lng_lo) & (lng <= lng_hi) & (lat >= lat_lo) & (lat <= lat_hi)
            if not valid.any():
                continue
            t, lng, lat = t[valid], lng[valid], lat[valid]
            xi = np.clip(np.searchsorted(lng_edges, lng, side='right') - 1, 0, gs - 1)
            yi = np.clip(np.searchsorted(lat_edges, lat, side='right') - 1, 0, gs - 1)
            h = t.dt.floor('h')
            tmp = pd.DataFrame({'h': h.values, 'gx': xi, 'gy': yi})
            agg = tmp.groupby(['h', 'gx', 'gy']).size().reset_index(name='count')
            for _, row in agg.iterrows():
                hour_grid[row['h']][int(row['gx']), int(row['gy'])] += row['count']
            n_processed += len(chunk)
            if n_processed % 5000000 == 0:
                log(f'  已处理 {n_processed/1e6:.0f}M 行...')

    log(f'处理完成: {n_processed} 行, {len(hour_grid)} 个小时')
    hours = sorted(hour_grid.keys())
    tensor = np.stack([hour_grid[h] for h in hours])

    grid_meta = {
        'grid_size': gs, 'lng_edges': lng_edges, 'lat_edges': lat_edges,
        'active_grids': set(),
        'lng_min': float(lng_edges[0]), 'lng_max': float(lng_edges[-1]),
        'lat_min': float(lat_edges[0]), 'lat_max': float(lat_edges[-1]),
    }
    nz = np.where(tensor.sum(axis=0) > 0)
    for x, y in zip(nz[0], nz[1]):
        grid_meta['active_grids'].add((int(x), int(y)))

    np.save(cache_tensor, tensor)
    np.save(cache_hours, np.array(hours))
    with open(cache_grid, 'wb') as f:
        _pk.dump(grid_meta, f)
    log(f'逐小时需求已缓存: {tensor.shape} -> {cache_tensor}')
    return tensor, np.array(hours), grid_meta


def load_weather_raw(hours, grid_meta=None):
    """加载与 hours 逐小时对齐的**原始**气象 (n, 4)。

    与旧实现的两点差异：
      1. 返回原始量（不再在这里做全量归一化）——归一化参数改由训练集估计，
         以满足「归一化统计量仅由训练集计算」的要求；
      2. 长度校验改为与 len(hours) 比对（旧实现拿完整时间跨度比对，等于没有校验）。
    """
    w_dir = os.path.join(config.PROCESSED_DIR, 'weather')
    os.makedirs(w_dir, exist_ok=True)
    w_feat_path = os.path.join(w_dir, 'weather_features.npy')
    w_start_path = os.path.join(w_dir, 'weather_start.txt')

    # 需求小时的完整跨度（可能含缺口），天气按**连续整点序列**缓存
    span_start = pd.Timestamp(hours[0]).floor('h')
    span_len = len(pd.date_range(span_start, pd.Timestamp(hours[-1]).floor('h'), freq='h'))
    cached_start = None
    if os.path.exists(w_start_path):
        try:
            cached_start = pd.Timestamp(open(w_start_path, encoding='utf-8').read().strip())
        except Exception:
            cached_start = None

    w_features = None
    if os.path.exists(w_feat_path):
        w_features = np.load(w_feat_path)
        ok = (len(w_features) >= span_len and cached_start is not None
              and cached_start == span_start)
        if not ok:
            log(f'天气缓存不匹配（起始 {cached_start} / 长度 {len(w_features)} vs '
                f'{span_start} / {span_len}），重新拉取')
            os.remove(w_feat_path)
            w_features = None
        else:
            log(f'加载天气特征(原始): {w_features.shape} 起始 {cached_start}')

    if w_features is None:
        if grid_meta is None:
            raise FileNotFoundError('无天气缓存且无 grid_meta，无法拉取天气')
        center_lat = (grid_meta['lat_min'] + grid_meta['lat_max']) / 2
        center_lng = (grid_meta['lng_min'] + grid_meta['lng_max']) / 2
        t_min = pd.Timestamp(hours[0]).strftime('%Y-%m-%d')
        t_max = pd.Timestamp(hours[-1]).strftime('%Y-%m-%d')
        log(f'拉取天气: ({center_lat:.2f}, {center_lng:.2f}) {t_min} ~ {t_max}')
        from src.weather import fetch_weather
        wdf = fetch_weather(center_lat, center_lng, t_min, t_max)
        wdf = wdf.drop_duplicates(subset='time').set_index('time').sort_index()
        hour_range = pd.date_range(pd.Timestamp(hours[0]).floor('h'),
                                   pd.Timestamp(hours[-1]).floor('h'), freq='h')
        wdf = wdf.reindex(hour_range).ffill().bfill()
        w_features = wdf[['temperature', 'precipitation', 'wind_speed', 'cloud_cover']] \
            .values.astype(np.float32)
        np.save(w_feat_path, w_features)
        with open(w_start_path, 'w', encoding='utf-8') as f:
            f.write(span_start.isoformat())
        log(f'天气特征已缓存: {w_features.shape} -> {w_feat_path}')

    # 按时间戳对齐到 hours（需求序列可能有缺口）
    t0 = span_start
    idx = ((pd.to_datetime(hours) - t0) / pd.Timedelta(hours=1)).astype(int).values
    idx = np.clip(idx, 0, len(w_features) - 1)
    return w_features[idx].astype(np.float32)


# ============================================================================
# 二、统一数据集（特征 / 基线 / 残差 / 切分）
# ============================================================================

def prepare_data(tensor=None, hours=None, grid_meta=None):
    """构建统一数据集：外生特征 z、条件基线 B、残差 r、切分索引与各统计量。"""
    if tensor is None:
        tensor, hours, grid_meta = build_hourly_demand()
    hours = pd.to_datetime(hours)
    n = len(hours)

    pool = config.XGB_POOL
    pg = config.GRID_SIZE // pool
    pooled = pool_3d_sum(tensor, pool)                       # (n, pg, pg)
    y = pooled.reshape(n, pg * pg).astype(np.float64)

    w_raw = load_weather_raw(hours, grid_meta)               # (n, 4) 原始气象
    dt_arr = day_type_array(hours)
    rb_arr = rain_bin(w_raw[:, 1])

    # ---- 时间切分（先切分，后统计，杜绝泄漏）----
    # 两级切分：先切留出段(holdout)，再在留出段内切验证集
    #   核心训练 76.5% | 验证 8.5% | 测试 15.0%（等效占比）
    n_tr = int(n * config.TRAIN_HOLDOUT_RATIO)
    n_test = n - n_tr
    n_val = max(int(n_tr * (1.0 - config.TRAIN_CORE_RATIO)), 1)
    n_core = n_tr - n_val
    idx_core = np.arange(0, n_core)
    idx_val = np.arange(n_core, n_tr)
    idx_test = np.arange(n_tr, n)

    # ---- 训练集统计量（仅 core 段）----
    w_mean = w_raw[idx_core].mean(axis=0)
    w_std = w_raw[idx_core].std(axis=0) + 1e-8
    w_norm = (w_raw - w_mean) / w_std

    y_mu = y[idx_core].mean(axis=0)
    y_sd = y[idx_core].std(axis=0) + 1e-8

    # ---- 条件基线：近期观测(W 小时滑窗) + 长期日期类型先验，逐级收缩 ----
    bl = ConditionalBaseline(pg, k_shrink=config.BASELINE_K,
                             window=config.BASELINE_LEVEL_WINDOW)
    bl.fit(pooled[idx_core], hours[idx_core], dt_arr[idx_core], rb_arr[idx_core])
    B_roll = bl.compute_rolling(pooled, hours, dt_arr, rb_arr,
                                freeze_idx=n_tr, deploy_idx=n)
    B = np.asarray(B_roll, dtype=np.float32).copy()
    # 测试段冻结：水平只用划分点之前的观测，测试段不参与基线校准
    B[n_tr:] = bl.transform(hours[n_tr:], dt_arr[n_tr:], rb_arr[n_tr:], which="eval")
    R = (pooled - B).astype(np.float32)

    r_mu = R[idx_core].reshape(n_core, -1).mean(axis=0)
    r_sd = R[idx_core].reshape(n_core, -1).std(axis=0) + 1e-8

    # ---- 外生特征 z ----
    z = exogenous_features(hours, weather_norm=w_norm)

    # ---- 静态空间先验图（区块长期水平，仅 core 段统计，未来任意日期可用）----
    static = np.log1p(np.clip(y_mu, 0, None))
    static = (static - static.mean()) / (static.std() + 1e-8)
    static_map = static.reshape(pg, pg).astype(np.float32)

    # ---- 情景廓线（仅 core 段统计）----
    core_mask = np.zeros(n, dtype=bool)
    core_mask[idx_core] = True
    profiles = build_scenario_profiles(hours, w_raw, train_mask=core_mask)
    save_scenario_profiles(profiles)
    # 覆盖 App 侧使用的归一化参数（训练集统计量）
    wdir = os.path.join(config.PROCESSED_DIR, 'weather')
    np.save(os.path.join(wdir, 'weather_mean.npy'), w_mean)
    np.save(os.path.join(wdir, 'weather_std.npy'), w_std)

    data = dict(
        hours=hours, pooled=pooled, y=y, w_raw=w_raw, w_norm=w_norm,
        z=z, dt_arr=dt_arr, rb_arr=rb_arr, B=B, R=R,
        pool=pool, pg=pg, n=n, n_tr=n_tr, n_val=n_val, n_core=n_core, n_test=n_test,
        idx_core=idx_core, idx_val=idx_val, idx_test=idx_test,
        y_mu=y_mu, y_sd=y_sd, r_mu=r_mu, r_sd=r_sd,
        w_mean=w_mean, w_std=w_std, static_map=static_map,
        baseline=bl, profiles=profiles,
    )
    log(f"数据集: 总 {n} 小时 | 训练(核心) {n_core} | 验证 {n_val} | 测试 {len(idx_test)}"
        f" | 等效占比 {n_core/n:.1%}/{n_val/n:.1%}/{len(idx_test)/n:.1%}")
    log(f"  基线 R²(相对零模型) = {1 - np.var(R[idx_core]) / np.var(y[idx_core]):.4f}")
    log(f"  残差/需求 std 比 = {R[idx_core].std() / y[idx_core].std():.1%}")
    log(f"  基线覆盖: {bl.coverage_report()}")
    return data


# ============================================================================
# 三、序列构造（残差序列 / 残差张量）
# ============================================================================

def build_sequences(z, R, r_mu, r_sd, static_map, indices, seq_len,
                    pg, use_history=True):
    """构造序列样本。

    每一步的特征 = [z(s) (N_FEATURES), r_norm(s) (pg²), mask(s) (1)]
      - s < 目标时刻：若 use_history 则为真实残差，mask=1
      - s = 目标时刻：残差位置填 0，mask=0（目标时刻的残差正是要预测的量）
    上下文 ctx = z(目标时刻)（供门控融合）
    输出 y_norm = (r(目标) − r_mu) / r_sd

    注：目标为**归一化残差**，只依赖 R 与外生特征 z；基线 B 与真实需求 y
    由调用方持有（B 在推理后加回，y 仅用于评估），故不作为参数传入。

    返回
      seq   (M, seq_len+1, N_FEATURES + pg² + 1)
      ctx   (M, N_FEATURES)
      grid  (M, seq_len+1, 3, pg, pg)  —— 通道 [r_norm, mask, 静态先验图]
      conds (M, seq_len+1, N_FEATURES)
      tgt   (M, pg²)
      索引元信息
    """
    P = pg * pg
    F = z.shape[1]
    T = seq_len + 1
    M = len(indices)
    seq = np.zeros((M, T, F + P + 1), dtype=np.float32)
    grid = np.zeros((M, T, 3, pg, pg), dtype=np.float32)
    conds = np.zeros((M, T, F), dtype=np.float32)
    ctx = np.zeros((M, F), dtype=np.float32)
    tgt = np.zeros((M, P), dtype=np.float32)

    for k, i in enumerate(indices):
        for j, s in enumerate(range(i - seq_len, i + 1)):
            sv = max(s, 0)
            seq[k, j, :F] = z[sv]
            conds[k, j] = z[sv]
            grid[k, j, 2] = static_map
            if s < i and s >= 0 and use_history:
                rn = (R[sv] - r_mu) / r_sd
                seq[k, j, F:F + P] = rn
                seq[k, j, F + P] = 1.0
                grid[k, j, 0] = rn.reshape(pg, pg)
                grid[k, j, 1] = 1.0
        ctx[k] = z[i]
        tgt[k] = (R[i] - r_mu) / r_sd
    return seq, ctx, grid, conds, tgt


# ============================================================================
# 四、模型定义（模块级，保证可 pickle / state_dict 可重建）
# ============================================================================

class MLPNet(nn.Module):
    """多层感知机：无序列机制的神经网络对照。"""

    def __init__(self, n_in, hidden, n_out, dropout):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_out)]
        self.net = nn.Sequential(*layers)
        self.cfg = dict(n_in=n_in, hidden=list(hidden), n_out=n_out, dropout=dropout)

    def forward(self, x):
        return self.net(x)


class SeqLSTMNet(nn.Module):
    """残差序列 LSTM（双通道）。

    通道一（外生直连）:  z(t) → MLP → r̂(z)，保证「无历史可用」时
                         模型至少不劣于纯外生映射（消融对照 MLP 的水平）；
    通道二（序列）    :  LSTM 循环 24h 残差序列 + 外生特征 → r̂(seq)

    输出为两条通道之和。这样 MLP 与 LSTM 的差异**只**来自序列通道，
    从而把「序列机制是否有贡献」与「网络是否难优化」两个因素解耦。
    """

    def __init__(self, n_step_feat, n_ctx, hidden, layers, dropout, n_out,
                 ctx_hidden=(128, 64)):
        super().__init__()
        self.lstm = nn.LSTM(n_step_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.seq_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out))
        layers_ctx, prev = [], n_ctx
        for h in ctx_hidden:
            layers_ctx += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers_ctx += [nn.Linear(prev, n_out)]
        self.ctx_net = nn.Sequential(*layers_ctx)
        self.cfg = dict(n_step_feat=n_step_feat, n_ctx=n_ctx, hidden=hidden,
                        layers=layers, dropout=dropout, n_out=n_out,
                        ctx_hidden=list(ctx_hidden))

    def forward(self, seq, ctx):
        out, _ = self.lstm(seq)
        h = out[:, -1]
        return self.seq_head(h) + self.ctx_net(ctx)


class PositionalEncoding(nn.Module):
    """正弦位置编码。

    自注意力本身对序列顺序无感，必须显式注入位置信息，否则「24 小时残差序列」
    会被当成无序集合，时间因果结构丢失。pe 注册为非持久化缓冲：
    它可由 d_model 与 max_len 完全重建，不进 state_dict，避免加载时形状不匹配。
    """

    def __init__(self, d_model, max_len=512, dropout=0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.max_len = int(max_len)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerNet(nn.Module):
    """残差序列 Transformer Encoder（双通道，与 SeqLSTMNet 同接口）。

    通道一（外生直连）:  z(t) → MLP → r̂(z)，保证「无历史可用」时
                         模型至少不劣于纯外生映射（冷启动不崩盘的关键）；
    通道二（序列）    :  n_step_feat → Linear(d_model) → 位置编码 →
                         TransformerEncoder → 取最后一步 → r̂(seq)

    输出为两通道之和。与 LSTM 的唯一差别是「循环单元 → 自注意力」：
    序列中任意两步直接可见，长程依赖不需逐步传递，因此对 24 小时窗口内
    相隔较远的时段（如早高峰与当前时刻）建模更直接。
    """

    def __init__(self, n_step_feat, n_ctx, d_model, nhead, num_layers, dropout,
                 n_out, ctx_hidden=(128, 64), dim_feedforward=128, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(n_step_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.seq_head = nn.Linear(d_model, n_out)
        layers_ctx, prev = [], n_ctx
        for h in ctx_hidden:
            layers_ctx += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers_ctx += [nn.Linear(prev, n_out)]
        self.ctx_net = nn.Sequential(*layers_ctx)
        self.cfg = dict(n_step_feat=n_step_feat, n_ctx=n_ctx, d_model=d_model,
                        nhead=nhead, num_layers=num_layers, dropout=dropout,
                        n_out=n_out, ctx_hidden=list(ctx_hidden),
                        dim_feedforward=dim_feedforward, max_len=int(max_len))

    def forward(self, seq, ctx):
        x = self.pos_enc(self.input_proj(seq))
        x = self.encoder(x)
        return self.seq_head(x[:, -1]) + self.ctx_net(ctx)


class FiLMConvLSTMCell(nn.Module):
    """带 FiLM 条件调制的 ConvLSTM 单元。

    FiLM: 由外生条件 z 生成 (γ, β)，对门控预激活做逐通道仿射调制，
    使「工作日/周末/假日/天气」显式地调节时空动态，
    而不是像原实现那样把条件广播成常量输入。
    """

    def __init__(self, in_ch, hid_ch, kernel_size, cond_dim):
        super().__init__()
        self.hidden_channels = hid_ch
        pad = kernel_size[0] // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=pad)
        self.film = nn.Linear(cond_dim, 8 * hid_ch)   # γ, β 各 4·hid_ch（对应四个门）
        self.hid_ch = hid_ch

    def forward(self, x, h_cur, cond):
        h, c = h_cur
        gates = self.conv(torch.cat([x, h], dim=1))
        gamma, beta = self.film(cond).view(cond.size(0), 8 * self.hid_ch, 1, 1) \
            .chunk(2, dim=1)
        gates = gates * (1.0 + gamma) + beta
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new


class SeqConvLSTMNet(nn.Module):
    """残差张量 ConvLSTM + FiLM 条件调制（双通道）。

    通道一（外生直连）:  z(t) → MLP → r̂(z)（5×5 展平），保证冷启动基线水平；
    通道二（时空）    :  在 5×5 残差图序列上卷积，外生条件经 FiLM 逐通道调制。

    隐状态初值可学习且空间可变，配合静态空间先验图，保证即使在无历史残差
    的冷启动下也能产生空间异质的输出，而不会退化为常量图。
    """

    def __init__(self, in_ch, cond_dim, hidden_channels, kernel_size, pg,
                 ctx_hidden=(128, 64)):
        super().__init__()
        self.pg = pg
        chs = [in_ch] + list(hidden_channels)
        self.cells = nn.ModuleList([
            FiLMConvLSTMCell(chs[i], chs[i + 1], kernel_size, cond_dim)
            for i in range(len(hidden_channels))
        ])
        self.out_conv = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)
        self.h0 = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, pg, pg)) for ch in hidden_channels])
        self.c0 = nn.ParameterList([
            nn.Parameter(torch.zeros(1, ch, pg, pg)) for ch in hidden_channels])
        layers_ctx, prev = [], cond_dim
        for h in ctx_hidden:
            layers_ctx += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers_ctx += [nn.Linear(prev, pg * pg)]
        self.ctx_net = nn.Sequential(*layers_ctx)
        self.cfg = dict(in_ch=in_ch, cond_dim=cond_dim,
                        hidden_channels=list(hidden_channels),
                        kernel_size=list(kernel_size), pg=pg,
                        ctx_hidden=list(ctx_hidden))

    def forward(self, grids, conds):
        B, T = grids.shape[0], grids.shape[1]
        H = W = self.pg
        h_states = [(self.h0[i].expand(B, -1, H, W).contiguous(),
                     self.c0[i].expand(B, -1, H, W).contiguous())
                    for i in range(len(self.cells))]
        for t in range(T):
            for i, cell in enumerate(self.cells):
                inp = grids[:, t] if i == 0 else h_states[i - 1][0]
                h_states[i] = cell(inp, h_states[i], conds[:, t])
        seq_out = self.out_conv(h_states[-1][0]).reshape(B, -1)
        return seq_out + self.ctx_net(conds[:, -1])


MLPNet.__module__ = _MODULE_NAME
SeqLSTMNet.__module__ = _MODULE_NAME
PositionalEncoding.__module__ = _MODULE_NAME
TransformerNet.__module__ = _MODULE_NAME
FiLMConvLSTMCell.__module__ = _MODULE_NAME
SeqConvLSTMNet.__module__ = _MODULE_NAME


# ============================================================================
# 五、训练：GBT
# ============================================================================

def _gbt_search(data, tag="GBT"):
    """在验证集上对 HistGBR 做小网格搜索（逐区块、目标 = 归一化残差）。

    由 config.GBT_SEARCH 控制是否执行：置 False 时直接返回缺省参数，
    跳过 225 次逐区块拟合（训练中最大的单项开销），用于调试与快速迭代。
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    if not getattr(config, "GBT_SEARCH", True):
        default = {k: config.XGB_PARAMS[k] for k in ("max_depth", "learning_rate", "max_iter")}
        log(f"  {tag} 跳过超参搜索（GBT_SEARCH=False），使用缺省参数 {default}")
        return default
    idx_core, idx_val = data["idx_core"], data["idx_val"]
    z = data["z"]
    tgt = ((data["R"].reshape(data["n"], -1) - data["r_mu"]) / data["r_sd"]).astype(np.float32)
    Xtr, Xva = z[idx_core], z[idx_val]
    ytr, yva = tgt[idx_core], tgt[idx_val]
    grid = config.GBT_GRID
    best = None
    for md in grid["max_depth"]:
        for lr in grid["learning_rate"]:
            for mi in grid["max_iter"]:
                preds = []
                for c in range(data["pg"] * data["pg"]):
                    m = HistGradientBoostingRegressor(
                        max_iter=mi, max_depth=md, learning_rate=lr,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=20, random_state=config.RANDOM_SEED)
                    m.fit(Xtr, ytr[:, c])
                    preds.append(m.predict(Xva))
                p = np.stack(preds, axis=-1)
                mse = float(np.mean((p - yva) ** 2))
                if best is None or mse < best[0]:
                    best = (mse, dict(max_depth=md, learning_rate=lr, max_iter=mi))
    log(f"  {tag} 超参搜索完成 → {best[1]} (验证 MSE={best[0]:.4f})")
    return best[1]


def _train_gbt(data):
    """梯度提升树：逐区块拟合归一化残差，特征仅 z(t)。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    idx_core = data["idx_core"]
    P = data["pg"] * data["pg"]
    tgt = ((data["R"].reshape(data["n"], -1) - data["r_mu"]) / data["r_sd"]).astype(np.float32)
    params = _gbt_search(data)
    Xtr = data["z"][idx_core]
    models = []
    for c in range(P):
        m = HistGradientBoostingRegressor(
            **params, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=config.RANDOM_SEED)
        m.fit(Xtr, tgt[idx_core][:, c])
        models.append(m)
    log(f"  GBT 已训练 {P} 个区块回归器 {params}")
    return {"type": "gbt", "models": models, "params": params}, tgt


# ============================================================================
# 六、训练：PyTorch 模型通用循环
# ============================================================================

def _train_torch(model, forward_fn, tr_tensors, va_tensors, params, tag,
                 history_dropout=0.3, drop_spec=None):
    """通用训练循环：逐 epoch 验证、早停、ReduceLROnPlateau、历史 dropout。

    forward_fn(model, batch) -> 预测 (B, P)
    batch 为元组；drop_spec = (批次元组下标, 需要置零的通道/列切片函数)
    """
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)

    tr = [t.to(DEVICE) if torch.is_tensor(t) else torch.as_tensor(t, device=DEVICE)
          for t in tr_tensors]
    va = [t.to(DEVICE) if torch.is_tensor(t) else torch.as_tensor(t, device=DEVICE)
          for t in va_tensors]

    n = tr[0].shape[0]
    batch_size = min(params.get("batch_size", 64), n)
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                           weight_decay=params.get("weight_decay", 0.0))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=params.get("lr_reduce_factor", 0.5),
        patience=params.get("lr_reduce_patience", 5))
    loss_fn = nn.MSELoss()

    y_idx = len(tr) - 1   # 约定：最后一维是目标

    def _forward_with_drop(batch, drop_p):
        batch = list(batch)
        if drop_p > 0 and drop_spec is not None:
            ti, masker = drop_spec
            b = batch[ti].shape[0]
            keep = (torch.rand(b, device=batch[ti].device) >= drop_p)
            batch[ti] = masker(batch[ti], keep)
        return forward_fn(model, tuple(batch))

    best, best_state, patience = float("inf"), None, 0
    for epoch in range(params["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = tuple(t[idx] for t in tr)
            opt.zero_grad()
            pred = _forward_with_drop(batch, history_dropout)
            loss = loss_fn(pred, batch[y_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        train_loss = tot / max(nb, 1)

        val_cold_batch = _cold_batch(va, drop_spec)
        model.eval()
        with torch.no_grad():
            val_cold = loss_fn(forward_fn(model, val_cold_batch), va[y_idx]).item()
            val_roll = loss_fn(forward_fn(model, va), va[y_idx]).item()
        val_loss = 0.5 * (val_cold + val_roll) if params.get("val_mode", "both") == "both" \
            else val_cold
        sched.step(val_loss)
        if val_loss < best - 1e-6:
            best, best_state, patience = val_loss, {k: v.detach().clone()
                                                    for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"  {tag} epoch {epoch+1}/{params['epochs']} train={train_loss:.5f} "
                f"val_cold={val_cold:.5f} val_roll={val_roll:.5f} "
                f"lr={opt.param_groups[0]['lr']:.2e}")
        if patience >= params.get("early_stop_patience", 25):
            log(f"  {tag} 早停 @ epoch {epoch+1}（最优验证 {best:.5f}）")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best


def _cold_batch(batch, drop_spec):
    """把一批样本转成冷启动形态。

    只丢弃**历史残差通道及其掩码**，外生特征 z 与静态空间先验必须保留——
    冷启动的含义是「没有历史需求观测」，而不是「没有日历与气象信息」。
    """
    if drop_spec is None:
        return tuple(batch)
    ti, masker = drop_spec
    out = list(batch)
    keep = torch.zeros(out[ti].shape[0], device=out[ti].device)
    out[ti] = masker(out[ti], keep)
    return tuple(out)


def _lstm_forward(model, batch):
    seq, ctx, _grid, _conds, _tgt = batch
    return model(seq, ctx)


def _lstm_drop(seq, keep):
    """按样本随机丢弃历史残差：残差列与掩码列同时置零。"""
    out = seq.clone()
    F = N_FEATURES
    P = out.shape[-1] - F - 1
    m = keep.view(-1, 1, 1).to(out.dtype)
    out[..., F:F + P] = out[..., F:F + P] * m
    out[..., F + P] = out[..., F + P] * m.view(-1, 1)
    return out


def _convlstm_forward(model, batch):
    _seq, _ctx, grid, conds, _tgt = batch
    return model(grid, conds)


def _convlstm_drop(grid, keep):
    out = grid.clone()
    m = keep.view(-1, 1, 1, 1).to(out.dtype)
    out[:, :, 0] = out[:, :, 0] * m
    out[:, :, 1] = out[:, :, 1] * m
    return out


def _train_mlp(data):
    tr, va = _make_mlp_tensors(data)
    p = config.MLP_PARAMS
    model = MLPNet(N_FEATURES, p["hidden"], data["pg"] * data["pg"], p["dropout"]).to(DEVICE)
    model, best = _train_torch(model, lambda m, b: m(b[0]), tr, va, p, "MLP",
                               history_dropout=0.0)
    log(f"  MLP 完成（验证 MSE={best:.5f}）")
    return {"type": "mlp", "model": model}, None


def _make_mlp_tensors(data):
    idx_tr, idx_va = data["idx_core"], data["idx_val"]
    tgt = ((data["R"].reshape(data["n"], -1) - data["r_mu"]) / data["r_sd"]).astype(np.float32)
    tr = (torch.FloatTensor(data["z"][idx_tr]), torch.FloatTensor(tgt[idx_tr]))
    va = (torch.FloatTensor(data["z"][idx_va]), torch.FloatTensor(tgt[idx_va]))
    return tr, va


def _transformer_forward(model, batch):
    """与 LSTM 同轨：输入同为 (seq, ctx)，历史残差通道的丢弃规则也相同。"""
    seq, ctx, _grid, _conds, _tgt = batch
    return model(seq, ctx)


def _train_transformer(data):
    p = config.TRANSFORMER_PARAMS
    seq_len = p["seq_len"]
    P = data["pg"] * data["pg"]
    tr, va = _seq_tensors(data, data["idx_core"], data["idx_val"], seq_len, data["pg"])
    n_step_feat = N_FEATURES + P + 1
    model = TransformerNet(n_step_feat, N_FEATURES, p["d_model"], p["nhead"],
                           p["num_layers"], p["dropout"], P,
                           p.get("ctx_hidden", (128, 64)),
                           p.get("dim_feedforward", 128)).to(DEVICE)
    model, best = _train_torch(model, _transformer_forward, tr, va, p, "Transformer",
                               history_dropout=p.get("history_dropout", 0.3),
                               drop_spec=(0, _lstm_drop))
    log(f"  Transformer 完成（验证 MSE={best:.5f}）")
    return {"type": "transformer", "model": model, "seq_len": seq_len}, None


def _train_lstm(data):
    p = config.LSTM_PARAMS
    seq_len = p["seq_len"]
    P = data["pg"] * data["pg"]
    tr, va = _seq_tensors(data, data["idx_core"], data["idx_val"], seq_len, data["pg"])
    n_step_feat = N_FEATURES + P + 1
    model = SeqLSTMNet(n_step_feat, N_FEATURES, p["hidden_size"], p["num_layers"],
                       p["dropout"], P, p.get("ctx_hidden", (128, 64))).to(DEVICE)
    model, best = _train_torch(model, _lstm_forward, tr, va, p, "LSTM",
                               history_dropout=p.get("history_dropout", 0.3),
                               drop_spec=(0, _lstm_drop))
    log(f"  LSTM 完成（验证 MSE={best:.5f}）")
    return {"type": "lstm", "model": model, "seq_len": seq_len}, None


def _train_convlstm(data):
    p = config.CONVLSTM_PARAMS
    seq_len = p["seq_len"]
    P = data["pg"] * data["pg"]
    tr, va = _seq_tensors(data, data["idx_core"], data["idx_val"], seq_len, data["pg"])
    model = SeqConvLSTMNet(3, N_FEATURES, p["hidden_channels"], p["kernel_size"],
                           data["pg"], p.get("ctx_hidden", (128, 64))).to(DEVICE)
    model, best = _train_torch(model, _convlstm_forward, tr, va, p, "ConvLSTM",
                               history_dropout=p.get("history_dropout", 0.3),
                               drop_spec=(2, _convlstm_drop))
    log(f"  ConvLSTM 完成（验证 MSE={best:.5f}）")
    return {"type": "convlstm", "model": model, "seq_len": seq_len}, None


def _seq_tensors(data, idx_tr, idx_va, seq_len, pg):
    common = dict(z=data["z"], R=data["R"].reshape(data["n"], -1),
                  r_mu=data["r_mu"], r_sd=data["r_sd"],
                  static_map=data["static_map"], seq_len=seq_len, pg=pg)
    out = []
    for idx in (idx_tr, idx_va):
        seq, ctx, grid, conds, tgt = build_sequences(indices=idx, **common)
        out.append((torch.FloatTensor(seq), torch.FloatTensor(ctx),
                    torch.FloatTensor(grid), torch.FloatTensor(conds),
                    torch.FloatTensor(tgt)))
    return out[0], out[1]


# ============================================================================
# 七、统一评估
# ============================================================================

def predict_all(data, results, indices, mode="cold", seq_len=None):
    """在给定索引集合上产出各模型的池化预测 (n, pg, pg)。

    mode: 'cold' 无历史残差 / 'rolling' 使用真实残差历史
    """
    n, pg = data["n"], data["pg"]
    P = pg * pg
    idx = np.asarray(indices)
    B = data["B"][idx]
    out = {}

    if "gbt" in results:
        r = results["gbt"]
        pred_norm = np.stack([m.predict(data["z"][idx]) for m in r["models"]], axis=-1)
        out["gbt"] = _invert(pred_norm, data, pg) + B
    if "mlp" in results:
        r = results["mlp"]
        m = r["model"].to(DEVICE).eval()
        with torch.no_grad():
            p = m(torch.FloatTensor(data["z"][idx]).to(DEVICE)).cpu().numpy()
        out["mlp"] = _invert(p, data, pg) + B
    for key, fwd in (("lstm", _lstm_forward),
                     ("convlstm", _convlstm_forward),
                     ("transformer", _transformer_forward)):
        if key not in results:
            continue
        r = results[key]
        sl = seq_len or r.get("seq_len", config.LSTM_PARAMS["seq_len"])
        seq, ctx, grid, conds, tgt = build_sequences(
            z=data["z"], R=data["R"].reshape(n, -1),
            r_mu=data["r_mu"], r_sd=data["r_sd"], static_map=data["static_map"],
            indices=idx, seq_len=sl, pg=pg, use_history=(mode == "rolling"))
        batch = [torch.FloatTensor(seq), torch.FloatTensor(ctx),
                 torch.FloatTensor(grid), torch.FloatTensor(conds),
                 torch.FloatTensor(tgt)]
        # 冷启动已由 build_sequences(use_history=False) 处理：
        # 残差通道与掩码为 0，外生特征 z 与静态先验图完整保留
        # （训练期的 history_dropout 走 _train_torch 的 drop_spec，推理期不需要）
        batch = [t.to(DEVICE) for t in batch]
        m = r["model"].to(DEVICE).eval()
        with torch.no_grad():
            p = fwd(m, tuple(batch)).cpu().numpy()
        out[key] = _invert(p, data, pg) + B
    return out


def _invert(pred_norm, data, pg):
    return (pred_norm * data["r_sd"] + data["r_mu"]).reshape(-1, pg, pg).astype(np.float32)


def evaluate_all(data, results, indices=None, tag=""):
    """对全部模型在**完全相同的索引集合**上评估，返回指标字典。"""
    idx = data["idx_test"] if indices is None else np.asarray(indices)
    y_true = data["pooled"][idx]
    hours = data["hours"][idx]
    dt_arr = data["dt_arr"][idx]
    rb_arr = data["rb_arr"][idx]

    metrics = {}
    for mode in ("cold", "rolling"):
        preds = predict_all(data, results, idx, mode=mode)
        for k, p in preds.items():
            m = core_metrics(y_true.ravel(), p.ravel())
            m["segments"] = segmented_metrics(y_true, p, hours, dt_arr, rb_arr)
            m["per_cell_r2_mean"] = float(np.nanmean(per_cell_r2(y_true, p)))
            m["mode"] = mode
            m["n_hours"] = int(len(idx))
            metrics[f"{k}|{mode}"] = m
            log(f"  [{tag}{mode}] {k:9s} {format_metrics(m)}")

    # 基线自身（零模型参考：ŷ = B）
    b = data["B"][idx]
    mb = core_metrics(y_true.ravel(), b.ravel())
    mb["segments"] = segmented_metrics(y_true, b, hours, dt_arr, rb_arr)
    mb["mode"] = "baseline"
    mb["n_hours"] = int(len(idx))
    metrics["baseline|baseline"] = mb
    log(f"  [{tag}baseline ] 条件基线   {format_metrics(mb)}")
    return metrics


# ============================================================================
# 八、训练主流程
# ============================================================================

def train(force=False, which=None, dump_arrays=False):
    which = which or ("gbt", "transformer", "lstm", "convlstm")
    if os.path.exists(MODEL_PATH) and not force:
        log(f"周期模型已存在: {MODEL_PATH}，跳过（--force 强制重训）")
        return MODEL_PATH

    log("=" * 66)
    log("训练：条件基线 + 残差学习（GBT / Transformer / LSTM / ConvLSTM）")
    log("=" * 66)
    t_start = time.time()
    tensor, hours, grid_meta = build_hourly_demand(force=False)
    data = prepare_data(tensor, hours, grid_meta)

    results, aux = {}, {}
    if "gbt" in which:
        log(">>> 训练 梯度提升树（传统机器学习）")
        results["gbt"], aux["gbt_tgt"] = _train_gbt(data)
    if "mlp" in which:
        log(">>> 训练 多层感知机（消融对照，无序列机制；可选，非默认）")
        results["mlp"], _ = _train_mlp(data)
    if "transformer" in which:
        log(">>> 训练 残差序列 Transformer Encoder（自注意力深度学习）")
        results["transformer"], _ = _train_transformer(data)
    if "lstm" in which:
        log(">>> 训练 残差序列 LSTM（时序深度学习）")
        results["lstm"], _ = _train_lstm(data)
    if "convlstm" in which:
        log(">>> 训练 残差张量 ConvLSTM + FiLM（时空深度学习）")
        results["convlstm"], _ = _train_convlstm(data)

    log(">>> 统一评估（测试集，同一小时集合）")
    metrics = evaluate_all(data, results)

    def _n_params(r):
        """GBT 记为「森林总树数」，神经网络记为可训练参数个数。

        取不到树数时返回 None（而非 -1），避免把哨兵值当成真实规模写进
        论文表格；meta["model_complexity_note"] 会记录受影响的模型。
        """
        if r["type"] == "gbt":
            total = 0
            for m in r["models"]:
                n_iter = getattr(m, "n_iter_", None)
                if n_iter is None:
                    return None
                total += int(np.sum(n_iter))
            return total
        return int(sum(p.numel() for p in r["model"].parameters()))

    complexity = {k: _n_params(v) for k, v in results.items()}
    unresolved = [k for k, v in complexity.items() if v is None]

    # 训练数据出处（来自数据集登记表）：写入 meta，供界面判断
    # 「当前模型是否仍是这批数据训出来的」，换数据后自动提示重训
    try:
        from src.dataset_registry import active_files, fingerprint, summarize
        _ds_files = active_files()
        _data_sources = {
            "fingerprint": fingerprint(_ds_files),
            "n_files": len(_ds_files),
            "total_bytes": sum(os.path.getsize(p) for p in _ds_files),
            "files": [{"name": os.path.basename(p),
                       "size": os.path.getsize(p)} for p in _ds_files],
            "summary": summarize(),
        }
    except Exception as e:            # 登记表不可用时不影响训练本身
        _data_sources = {"error": str(e)}

    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": _data_sources,
        "data_range": [str(pd.Timestamp(hours.min()).date()),
                       str(pd.Timestamp(hours.max()).date())],
        "n_hours": len(hours),
        "grid_size": config.GRID_SIZE,
        "pool": config.XGB_POOL,
        "pool_mode": "sum",
        "feature_names": FEATURE_NAMES,
        "architecture": "条件基线 + 残差学习",
        "model_complexity": complexity,
        "model_complexity_note": ("未取到树数（记为 null）: " + ", ".join(unresolved)
                                  if unresolved else "全部模型规模均可量化"),
        # GBT 验证集选出的超参。落库以便论文与界面可直接引用，
        # 无需回读 results["gbt"]["params"]。GBT_SEARCH=False 时为缺省参数，
        # 同时用 gbt_search_enabled 明确标注该结果是否经过搜索。
        "gbt_best_params": (results.get("gbt") or {}).get("params"),
        "gbt_search_enabled": bool(getattr(config, "GBT_SEARCH", True)),
        "split": {"n": data["n"], "n_core": data["n_core"], "n_val": data["n_val"],
                  "n_test": int(len(data["idx_test"])),
                  "holdout_ratio": config.TRAIN_HOLDOUT_RATIO,
                  "core_ratio_in_holdout": config.TRAIN_CORE_RATIO,
                  "effective_ratio": {"core": round(data["n_core"] / data["n"], 4),
                                      "val": round(data["n_val"] / data["n"], 4),
                                      "test": round(data["n_test"] / data["n"], 4)}},
        "metrics": metrics,
        "baseline_coverage": data["baseline"].coverage_report(),
        "scenario_profiles": list(WEATHER_SCENARIOS),
        "models_trained": list(results.keys()),
        "suggest_retrain_days": config.SUGGEST_RETRAIN_DAYS,
        "notes": ("冷启动=无历史残差（纯未来预测，App 默认口径）；"
                  "滚动=提供前 24h 真实残差（短期滚动预测）。"
                  "MAPE 仅在真实值≥1 的样本上计算。"),
    }
    _save_results(data, results, meta, with_arrays=dump_arrays)
    log(f"模型已保存（总耗时 {time.time()-t_start:.0f}s）")
    return MODEL_PATH


def _save_results(data, results, meta, with_arrays=False):
    """保存：概率模型与统计量进 pkl，神经网络权重进 .pt。

    pkl 默认**只含推理必需项**（stat / meta / pool / results），体积约 1 MB。
    中间大数组（pooled / hours / y / B / R）合计上百 MB，仅用于离线复算，
    默认不落盘——这些量随时可由 `prepare_data` 从缓存重建，属派生数据。
    需要时用 `--dump-arrays` 显式写出。
    """
    stat = {
        "baseline": data["baseline"].to_dict(),
        "y_mu": data["y_mu"], "y_sd": data["y_sd"],
        "r_mu": data["r_mu"], "r_sd": data["r_sd"],
        "w_mean": data["w_mean"], "w_std": data["w_std"],
        "static_map": data["static_map"],
        "pg": data["pg"], "pool": data["pool"],
        "n": data["n"], "n_tr": data["n_tr"], "n_val": data["n_val"],
    }
    payload = {
        "stat": stat, "meta": meta,
        "pool": data["pool"],
        "results": {},
    }
    if with_arrays:
        payload.update({"pooled": data["pooled"], "hours": data["hours"],
                        "y": data["y"], "B": data["B"], "R": data["R"]})
    if "gbt" in results:
        payload["results"]["gbt"] = results["gbt"]
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    log(f"已保存主模型文件: {MODEL_PATH}（{size_mb:.2f} MB"
        f"{'，含中间数组' if with_arrays else '，不含中间数组'}）")

    # mlp 仍在此列：权重文件保留，任何时候都载入，作为内部消融对照的活证据
    for key in ("mlp", "lstm", "convlstm", "transformer"):
        r = results.get(key)
        if r is None:
            continue
        path = MODEL_PATH.replace(".pkl", f"_{key}.pt")
        torch.save({"model": r["model"].state_dict(), "type": r["type"],
                    "cfg": r["model"].cfg, "meta": meta, "pool": data["pool"],
                    "pg": data["pg"], "seq_len": r.get("seq_len", 24),
                    "r_mu": data["r_mu"], "r_sd": data["r_sd"]}, path)
        log(f"已保存 {key} 权重: {path}")


def load_models():
    """加载全部已训练模型与统计量；主文件缺失返回 None。

    stat 做形状/类型校验，避免权重与统计量错配后静默产出错误预测。
    """
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    results = dict(payload.get("results", {}))
    stat = payload["stat"]
    pool = payload.get("pool", config.XGB_POOL)
    pg = stat["pg"]

    # ---- stat 校验（权重与统计量必须同源）----
    required = ("baseline", "y_mu", "y_sd", "r_mu", "r_sd",
                "w_mean", "w_std", "static_map", "pg", "pool")
    missing = [k for k in required if k not in stat]
    if missing:
        raise KeyError(f"模型文件 stat 字段缺失: {missing}；请重新训练")
    if not isinstance(stat["baseline"], dict):
        raise TypeError(f"stat['baseline'] 应为 dict，实际 {type(stat['baseline']).__name__}")
    sm = np.asarray(stat["static_map"], dtype=np.float64)
    if sm.shape != (pg, pg):
        raise ValueError(f"stat['static_map'] 形状应为 ({pg},{pg})，实际 {sm.shape}")
    for k in ("r_mu", "r_sd"):
        if np.asarray(stat[k]).size != pg * pg:
            raise ValueError(f"stat['{k}'] 长度应为 {pg * pg}，实际 {np.asarray(stat[k]).size}")
    if not np.all(np.isfinite(sm)):
        raise ValueError("stat['static_map'] 含 NaN/Inf")

    for key, cls in (("mlp", MLPNet), ("lstm", SeqLSTMNet),
                     ("convlstm", SeqConvLSTMNet), ("transformer", TransformerNet)):
        path = MODEL_PATH.replace(".pkl", f"_{key}.pt")
        if not os.path.exists(path):
            continue
        sd = torch.load(path, map_location="cpu", weights_only=False)
        cfg = sd["cfg"]
        if key == "mlp":
            m = MLPNet(cfg["n_in"], cfg["hidden"], cfg["n_out"], cfg["dropout"])
        elif key == "lstm":
            m = SeqLSTMNet(cfg["n_step_feat"], cfg["n_ctx"], cfg["hidden"],
                           cfg["layers"], cfg["dropout"], cfg["n_out"],
                           cfg.get("ctx_hidden", (128, 64)))
        elif key == "transformer":
            m = TransformerNet(cfg["n_step_feat"], cfg["n_ctx"], cfg["d_model"],
                               cfg["nhead"], cfg["num_layers"], cfg["dropout"],
                               cfg["n_out"], cfg.get("ctx_hidden", (128, 64)),
                               cfg.get("dim_feedforward", 128),
                               cfg.get("max_len", 512))
        else:
            m = SeqConvLSTMNet(cfg["in_ch"], cfg["cond_dim"],
                               cfg["hidden_channels"], cfg["kernel_size"], cfg["pg"],
                               cfg.get("ctx_hidden", (128, 64)))
        m.load_state_dict(sd["model"])
        m.eval()
        results[key] = {"type": key, "model": m, "seq_len": sd.get("seq_len", 24)}
        log(f"已加载 {key} 权重")

    baseline = ConditionalBaseline.from_dict(stat["baseline"])
    return {"results": results, "meta": payload["meta"], "stat": stat,
            "baseline": baseline, "pool": pool, "pg": pg,
            "pooled": payload.get("pooled"), "hours": payload.get("hours")}

# ============================================================================
# 九、预测接口（供 App 调用）
# ============================================================================

# ----------------------------------------------------------------------------
# 滚动口径：历史残差上下文（推理期只读缓存）
# ----------------------------------------------------------------------------
_HIST_CACHE = {}
_HIST_LOCK = threading.Lock()


def build_history_cache(models=None, force=False):
    """构建（进程内缓存）推理期历史序列：hours / 真实需求 / 基线 / 残差 / 真实气象。

    口径与训练评估**逐元素同源**，这是滚动口径可复现的前提：
      * 划分点之前 → 因果滑窗基线 compute_rolling（窗口 [i−W, i)，不含当小时）
      * 划分点及之后 → 冻结在划分点的 eval 快照（从不接触留出段观测）
    因此 R = pooled − B 与 evaluate_all(mode="rolling") 使用的残差完全一致，
    界面上的滚动结果可与论文指标直接对照。

    只做内存缓存、不落盘：重训后模型文件变更，落盘缓存会与权重版本错配；
    重新构建成本约 0.2 s、占用约 1.4 MB，不值得冒这个风险。
    """
    if models is None:
        models = load_models()
    if models is None:
        return None
    ent = _HIST_CACHE.get("hist")
    if ent is not None and not force and ent["models"] is models:
        return ent

    with _HIST_LOCK:
        # 双检：并发会话可能同时构建，先到者生效
        ent = _HIST_CACHE.get("hist")
        if ent is not None and not force and ent["models"] is models:
            return ent

        stat = models["stat"]
        pg, pool = models["pg"], models["pool"]
        tensor, hours, grid_meta = build_hourly_demand()
        hours = pd.DatetimeIndex(hours)
        n = len(hours)
        n_tr = int(stat["n_tr"])

        pooled = pool_3d_sum(tensor, pool).astype(np.float64)
        w_raw = load_weather_raw(hours, grid_meta)
        dt_arr = day_type_array(hours)
        rb_arr = rain_bin(w_raw[:, 1])

        bl = models["baseline"]
        B = np.asarray(bl.compute_rolling(pooled, hours, dt_arr, rb_arr,
                                         freeze_idx=n_tr, deploy_idx=n),
                       dtype=np.float32).copy()
        B[n_tr:] = bl.transform(hours[n_tr:], dt_arr[n_tr:], rb_arr[n_tr:],
                                which="eval")
        R = (pooled.astype(np.float32) - B).astype(np.float32)
        w_norm = ((w_raw - stat["w_mean"]) / stat["w_std"]).astype(np.float32)

        ent = {
            "models": models, "hours": hours, "pooled": pooled.astype(np.float32),
            "B": B, "R": R, "w_norm": w_norm, "w_raw": w_raw,
            "pg": pg, "pool": pool, "n": n, "n_tr": n_tr,
            "pos": {ts: i for i, ts in enumerate(hours)},
        }
        _HIST_CACHE["hist"] = ent
        log(f"滚动口径历史缓存: {R.shape} | 可复现区间 {hours[n_tr]} ~ {hours[-1]}")
        return ent


def _resolve_seq_len(models, model_key, seq_len=None):
    """序列长度：显式入参 > 模型自带 cfg > 配置默认值（不写死 24）。"""
    if seq_len:
        return int(seq_len)
    r = (models or {}).get("results", {}).get(model_key) or {}
    if r.get("seq_len"):
        return int(r["seq_len"])
    return int(config.LSTM_PARAMS["seq_len"])


def get_rolling_context(target_ts, model_key="convlstm", models=None, seq_len=None):
    """目标时刻的滚动口径推理上下文；历史不足返回 None。

    返回 dict：
      residual      (seq_len, pg, pg) float32  目标时刻前 seq_len 小时的真实残差
      weather_norm  (seq_len+1, 4)   float32  与之对齐的真实气象（含目标时刻本身）
      hours         (seq_len+1,)              窗口对应的真实时刻（数据有缺口时与整点不同）
      precip_mm     目标时刻的真实降水量（用于条件基线的降雨分档）
      seq_len / start / end / pos / has_history_channel

    「历史不足」包含两种情况：目标时刻不在历史数据中（如未来日期）、
    或位于序列起点附近、前 seq_len 小时不完整。
    """
    hc = build_history_cache(models)
    if hc is None:
        return None
    ts = pd.Timestamp(target_ts)
    ts = ts.floor("h")
    i = hc["pos"].get(ts)
    if i is None:
        return None
    sl = _resolve_seq_len(hc["models"], model_key, seq_len)
    if i - sl < 0:
        return None
    lo, hi = i - sl, i + 1
    return {
        "seq_len": sl,
        "residual": hc["R"][lo:i].copy(),                 # (sl, pg, pg)
        "weather_norm": hc["w_norm"][lo:hi].copy(),        # (sl+1, 4)
        "hours": hc["hours"][lo:hi].values,                # (sl+1,) 窗口真实时刻
        "precip_mm": float(hc["w_raw"][i, 1]),
        "start": hc["hours"][lo], "end": hc["hours"][i - 1],
        "target": hc["hours"][i], "pos": int(i),
        "has_history_channel": model_key in ("lstm", "convlstm", "transformer"),
    }


def get_recent_residual(target_ts, model_key="convlstm", models=None, seq_len=None):
    """目标时刻前 seq_len 小时的真实需求残差 (seq_len, pg, pg)。

    残差 = 真实池化需求 − 条件基线，与训练/评估的口径逐元素一致。
    历史数据不足（未来日期、序列起点附近）时返回 None —— 调用方应据此提示
    用户改用冷启动口径，而不是静默降级。
    """
    ctx = get_rolling_context(target_ts, model_key, models, seq_len)
    return None if ctx is None else ctx["residual"]


def get_rolling_range(models=None):
    """滚动口径可复现区间 (start_ts, end_ts)。

    取留出测试段：该段既未参与训练，基线又冻结在划分点之前，
    是唯一能同时保证「无未来信息泄漏」与「与论文滚动指标同口径」的区间。
    模型不可用时返回 (None, None)。
    """
    hc = build_history_cache(models)
    if hc is None:
        return (None, None)
    return (hc["hours"][hc["n_tr"]], hc["hours"][-1])


def _baseline_at(models, ts, weather_scenario, holiday_override, profiles,
                 which="deploy", precip_mm=None):
    """目标小时的条件基线。

    which     : 'deploy'（截至最新观测的快照，纯未来预测用）
                'eval'（冻结在划分点，滚动口径必须用这个：与残差构造保持同一快照，
                否则 B 与 r 错配会产生系统性偏移）
    precip_mm : 给定则用该真实降水量定降雨档（滚动口径），否则按情景廓线取
    """
    precip = (_precip_of(weather_scenario, ts, profiles)
              if precip_mm is None else float(precip_mm))
    dt_arr = None
    if holiday_override is not None:
        dt_arr = np.array([2 if holiday_override else 0], dtype=np.int64)
    return models["baseline"].transform([ts], dt_arr=dt_arr,
                                        rb_arr=rain_bin([precip]), which=which)[0]



def _tensor(x):
    """统一转 float32 张量：容忍 list / 非 float64 的 ndarray。"""
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)


def predict_hour(target_date, hour, weather_scenario="晴天", event=None,
                 model_key="gbt", models=None, holiday_override=None,
                 recent_residual=None, rolling_weather=None,
                 baseline_which="deploy"):
    """预测目标日期某一小时的需求 (gs, gs)。

    target_date : 'YYYY-MM-DD'（任意未来日期）
    recent_residual : 可选，(seq_len, pg, pg) 的历史残差；给定时启用滚动口径，
                      否则为冷启动口径（App 默认，纯未来预测）
    rolling_weather : 可选，滚动口径的真实气象上下文
                      {"norm": (seq_len+1, 4) 已按训练集统计量归一化,
                       "precip_mm": float 目标小时真实降水量,
                       "hours": (seq_len+1,) 窗口真实时刻（可选，用于对齐数据缺口）}
                      给定时外生特征与基线降雨档均取真实历史值，
                      weather_scenario 的情景假设不再生效
    baseline_which : 'deploy'（默认）或 'eval'；滚动口径必须用 'eval'，
                      与 get_rolling_context 构造残差所用的快照保持一致
    """
    if models is None:
        models = load_models()
    if models is None or model_key not in models["results"]:
        return None

    stat = models["stat"]
    pool = models["pool"]
    gs = models["meta"]["grid_size"]
    pg = models["pg"]
    r = models["results"][model_key]
    ts = pd.Timestamp(f"{target_date} {hour:02d}:00:00")

    from src.date_features import load_scenario_profiles
    profiles = load_scenario_profiles()

    wnorm = None
    precip_mm = None
    if rolling_weather:
        wnorm = np.asarray(rolling_weather["norm"], dtype=np.float32)
        pm = rolling_weather.get("precip_mm")
        if pm is not None and np.isfinite(pm):
            precip_mm = float(pm)

    B = _baseline_at(models, ts, weather_scenario, holiday_override, profiles,
                     which=baseline_which, precip_mm=precip_mm)
    P = pg * pg

    if r["type"] in ("gbt", "mlp"):
        # 无序列结构：只取目标小时的（真实）气象，历史残差通道不存在
        wn = None if wnorm is None else wnorm[-1:]
        z = exogenous_features([ts], weather_norm=wn,
                               scenario=None if wn is not None else weather_scenario,
                               holiday_override=holiday_override, profiles=profiles)
        z = np.asarray(z, dtype=np.float32)
        if r["type"] == "gbt":
            pred_norm = np.array([m.predict(z)[0] for m in r["models"]])
        else:
            m = r["model"].to(DEVICE).eval()
            with torch.no_grad():
                pred_norm = m(_tensor(z).to(DEVICE)).cpu().numpy()[0]
        pred_p = (pred_norm * stat["r_sd"] + stat["r_mu"]).reshape(pg, pg) + B

    elif r["type"] in ("lstm", "convlstm", "transformer"):
        seq_len = r.get("seq_len") or config.LSTM_PARAMS["seq_len"]
        # 序列窗口必须取「残差对应的那几行真实时刻」而非 ts 往前推的整点序列：
        # 逐小时需求存在缺口（本数据集 6 个缺失小时），按整点铺开会把外生特征错位，
        # 与训练/评估的 build_sequences(z[行窗口]) 口径不一致。
        win_hours = (rolling_weather or {}).get("hours")
        if win_hours is not None:
            steps = pd.DatetimeIndex(win_hours)
            if len(steps) != seq_len + 1:
                raise ValueError(f"rolling_weather['hours'] 长度应为 {seq_len + 1}，"
                                 f"实际 {len(steps)}")
        else:
            steps = pd.date_range(end=ts, periods=seq_len + 1, freq="h")
        z_seq = np.asarray(exogenous_features(
            steps,
            weather_norm=wnorm if wnorm is not None else None,
            scenario=None if wnorm is not None else weather_scenario,
            holiday_override=holiday_override, profiles=profiles), dtype=np.float32)
        if z_seq.ndim != 2 or z_seq.shape[1] != N_FEATURES or len(z_seq) != seq_len + 1:
            raise ValueError(f"外生特征形状异常: {z_seq.shape}，"
                             f"期望 ({seq_len + 1}, {N_FEATURES})")
        if wnorm is not None and wnorm.shape != (seq_len + 1, 4):
            raise ValueError(f"rolling_weather['norm'] 形状应为 ({seq_len + 1}, 4)，"
                             f"实际 {wnorm.shape}；与模型 seq_len 不匹配")
        use_hist = recent_residual is not None
        if use_hist:
            rr = np.asarray(recent_residual, dtype=np.float32)
            if rr.shape != (seq_len, pg, pg):
                raise ValueError(f"recent_residual 形状应为 ({seq_len}, {pg}, {pg})，"
                                 f"实际 {rr.shape}")
            rr = rr.reshape(seq_len, P)
            rr = (rr - stat["r_mu"]) / stat["r_sd"]
        seq = np.zeros((1, seq_len + 1, N_FEATURES + P + 1), dtype=np.float32)
        grid = np.zeros((1, seq_len + 1, 3, pg, pg), dtype=np.float32)
        conds = np.zeros((1, seq_len + 1, N_FEATURES), dtype=np.float32)
        for j in range(seq_len + 1):
            seq[0, j, :N_FEATURES] = z_seq[j]
            conds[0, j] = z_seq[j]
            grid[0, j, 2] = stat["static_map"]
            if use_hist and j < seq_len:
                seq[0, j, N_FEATURES:N_FEATURES + P] = rr[j]
                seq[0, j, N_FEATURES + P] = 1.0
                grid[0, j, 0] = rr[j].reshape(pg, pg)
                grid[0, j, 1] = 1.0
        m = r["model"].to(DEVICE).eval()
        with torch.no_grad():
            if r["type"] in ("lstm", "transformer"):
                out = m(_tensor(seq).to(DEVICE), _tensor(z_seq[-1:]).to(DEVICE))
            else:
                out = m(_tensor(grid).to(DEVICE), _tensor(conds).to(DEVICE))
        p = out.cpu().numpy()[0]
        pred_p = (p * stat["r_sd"] + stat["r_mu"]).reshape(pg, pg) + B
    else:
        return None

    # 求和池化还原：块状复制，避免平滑削峰、保持总量守恒
    pred = np.repeat(np.repeat(pred_p, pool, axis=0), pool, axis=1)
    pred = pred / (pool * pool)
    pred = np.clip(pred, 0, None)

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
            pred[mask] = pred[mask] * max(0.0, 1.0 - strength)
        pred = np.clip(pred, 0, None)
    return pred


def _precip_of(scenario, ts, profiles=None):
    """情景在该小时对应的原始降水量（用于条件基线的降雨分档）。"""
    w = scenario_hourly_weather(scenario, [ts], profiles)
    return float(w[0, 1])


def predict_day(target_date, weather_scenario="晴天", event=None, model_key="gbt",
                models=None, holiday_override=None, rolling=False):
    """预测目标日期全天 24 小时 → (24, gs, gs)。

    rolling=True 时逐小时取「前 seq_len 小时真实残差 + 真实气象」，
    基线同步切到 eval 快照。历史不足的小时（目标日期早于数据起点、
    或模型无序列结构）自动回退冷启动，不会抛错中断整条曲线。
    """
    if models is None:
        models = load_models()
    if models is None:
        return None
    out = []
    for h in range(24):
        rr = rw = None
        if rolling:
            ctx = get_rolling_context(f"{target_date} {h:02d}:00:00",
                                      model_key, models)
            if ctx is not None and ctx["has_history_channel"]:
                rr = ctx["residual"]
                rw = {"norm": ctx["weather_norm"], "precip_mm": ctx["precip_mm"],
                      "hours": ctx["hours"]}
        out.append(predict_hour(
            target_date, h, weather_scenario, event, model_key, models,
            holiday_override, recent_residual=rr, rolling_weather=rw,
            baseline_which="eval" if rr is not None else "deploy"))
    return np.stack(out)



def get_model_info():
    models = load_models()
    return None if models is None else models["meta"]


def get_metrics_table(models=None):
    """返回 (模型, 口径) → 指标的扁平表，便于论文出表。"""
    models = models or load_models()
    if models is None:
        return []
    rows = []
    for key, m in models["meta"]["metrics"].items():
        name, mode = key.split("|")
        rows.append({"模型": name, "口径": mode, **{k: v for k, v in m.items()
                                                    if not isinstance(v, dict)}})
    return rows


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重训")
    parser.add_argument("--models", default="gbt,transformer,lstm,convlstm",
                        help="逗号分隔模型列表；加 mlp 可一并重训消融对照")
    parser.add_argument("--dump-arrays", action="store_true",
                        help="额外把 pooled/hours/y/B/R 中间数组写进 pkl（体积大，仅离线复算用）")
    args = parser.parse_args()

    train(force=args.force, which=[m.strip() for m in args.models.split(",") if m.strip()],
          dump_arrays=args.dump_arrays)

    models = load_models()
    meta = models["meta"]
    print("\n" + "=" * 66)
    print("模型信息")
    print("=" * 66)
    print(f"  训练时间: {meta['trained_at']}")
    print(f"  数据范围: {meta['data_range'][0]} ~ {meta['data_range'][1]}")
    print(f"  切分: {meta['split']}")
    print(f"  基线覆盖: {meta['baseline_coverage']}")
    for key, m in meta["metrics"].items():
        print(f"  {key:20s} {format_metrics(m)}")

    print("\n--- 冷启动示例（纯未来预测） ---")
    for date, w in [("2021-05-01", "晴天"), ("2021-07-05", "晴天"), ("2026-08-15", "大雨")]:
        for mk in ["gbt", "mlp", "lstm", "convlstm"]:
            pred = predict_hour(date, 8, w, model_key=mk)
            if pred is not None:
                print(f"  {date} 08:00 {w} [{mk:8s}]: 总需求 {pred.sum():>10.0f} 辆 | "
                      f"最大网格 {pred.max():>7.1f}")
        print()
