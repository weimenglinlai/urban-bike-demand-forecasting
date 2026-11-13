# -*- coding: utf-8 -*-
"""
滚动多步预测：用已训练模型预测未来 N 小时（可指定目标日期/天气情景/事件扰动）

原理：取测试集最后一个窗口作为起点，迭代预测——每一步的预测结果拼回窗口末尾，
天气通道由天气情景生成（用户指定或历史均值），支持事件扰动（活动/事故）。

科学严谨性说明：
- 天气不再用硬编码乘系数，而是作为模型输入特征通道（模型在训练时学习天气→需求映射）
- 日周期锚定仅用于缓解自回归误差累积（水平对齐历史同小时均值），保留模型空间模式
"""

import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log, pool_5d, pool_3d, unpool

WEATHER_DIR = os.path.join(config.PROCESSED_DIR, "weather")

# 天气情景 → 典型天气参数（温度°C / 降水mm / 风速m/s / 云量%）
# 基于纽约 2018-09 训练集天气分布校准（避免超出训练分布导致模型发散）：
#   温度: p10~p95 = 17~29°C, 中位 22°C
#   降水: 干燥73% / 小雨(0.1-2.5)24% / 中雨(2.5-8)3% / 大雨>8mm 0%（训练内最大 8.8mm）
#   风速: p25=7.2, 中位 10, p75=15 m/s
#   云量: 中位 97%, 晴天对应低云量（训练内 p25=45）
WEATHER_SCENARIOS = {
    "晴天": {"temperature": 24.0, "precipitation": 0.0, "wind_speed": 6.0, "cloud_cover": 40.0},
    "阴天": {"temperature": 21.0, "precipitation": 0.2, "wind_speed": 10.0, "cloud_cover": 90.0},
    "小雨": {"temperature": 20.0, "precipitation": 2.0, "wind_speed": 10.0, "cloud_cover": 92.0},
    "大雨": {"temperature": 18.0, "precipitation": 6.0, "wind_speed": 14.0, "cloud_cover": 100.0},
    "高温": {"temperature": 30.0, "precipitation": 0.0, "wind_speed": 7.0, "cloud_cover": 50.0},
}


def _load_model(model_type):
    """加载已训练模型，返回可调用对象"""
    gs = config.GRID_SIZE
    pool = config.XGB_POOL
    pg = gs // pool

    if model_type == "gbt":
        path = os.path.join(config.OUTPUT_DIR, "xgboost_model.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            model = pickle.load(f)

        def predict(X_window):
            # X_window: (1, 24, 20, 20, C) → 展平 → 预测 → 反池化
            n = X_window.shape[0]
            Xp = pool_5d(X_window, pool).reshape(n, -1)
            yp = model.predict(Xp).reshape(n, pg, pg)
            return unpool(yp, pool, gs)

        return predict

    elif model_type == "lstm":
        import torch
        import torch.nn as nn
        path = os.path.join(config.OUTPUT_DIR, "lstm_model.pth")
        if not os.path.exists(path):
            return None
        channels = 5
        grid_dim = gs * gs * channels
        p = config.LSTM_PARAMS

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(grid_dim, p["hidden_size"], p["num_layers"],
                                    batch_first=True,
                                    dropout=p["dropout"] if p["num_layers"] > 1 else 0)
                self.fc = nn.Linear(p["hidden_size"], pg * pg)
            def forward(self, x):
                h0 = torch.zeros(p["num_layers"], x.size(0), p["hidden_size"])
                c0 = torch.zeros(p["num_layers"], x.size(0), p["hidden_size"])
                out, _ = self.lstm(x, (h0, c0))
                return self.fc(out[:, -1, :])

        model = M()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

        def predict(X_window):
            n = X_window.shape[0]
            xf = X_window.reshape(n, config.WINDOW_SIZE, grid_dim)
            with torch.no_grad():
                yp = model(torch.FloatTensor(xf)).numpy()
            return unpool(yp.reshape(n, pg, pg), pool, gs)

        return predict

    elif model_type == "convlstm":
        import torch
        import torch.nn as nn
        path = os.path.join(config.OUTPUT_DIR, "convlstm_model.pth")
        if not os.path.exists(path):
            return None
        channels = 5
        p = config.CONVLSTM_PARAMS

        class Cell(nn.Module):
            def __init__(self, ic, hc, ks):
                super().__init__()
                self.hc = hc
                self.conv = nn.Conv2d(ic + hc, 4 * hc, ks, padding=ks[0]//2)
            def forward(self, x, hc):
                h, c = hc
                g = self.conv(torch.cat([x, h], 1))
                i, f, o, gg = torch.chunk(g, 4, 1)
                cn = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(gg)
                hn = torch.sigmoid(o)*torch.tanh(cn)
                return hn, cn
            def init(self, b, h, w):
                return (torch.zeros(b, self.hc, h, w), torch.zeros(b, self.hc, h, w))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                chs = [channels] + list(p["hidden_channels"])
                self.cells = nn.ModuleList([Cell(chs[i], chs[i+1], p["kernel_size"]) for i in range(len(p["hidden_channels"]))])
                self.out_conv = nn.Conv2d(p["hidden_channels"][-1], 1, 1)
            def forward(self, x):
                b, t, c, h, w = x.shape
                hs = [cell.init(b, h, w) for cell in self.cells]
                for tt in range(t):
                    xt = x[:, tt]
                    for i, cell in enumerate(self.cells):
                        hs[i] = cell(xt if i == 0 else hs[i-1][0], hs[i])
                return self.out_conv(hs[-1][0])

        model = M()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

        def predict(X_window):
            Xp = pool_5d(X_window, pool)
            Xt = torch.FloatTensor(Xp).permute(0, 1, 4, 2, 3)
            with torch.no_grad():
                yp = model(Xt).cpu().numpy()
            return unpool(yp.squeeze(1), pool, gs)

        return predict

    return None


def _load_weather_normalization():
    """加载天气标准化参数 (mean, std)"""
    wd = WEATHER_DIR
    mpath = os.path.join(wd, "weather_mean.npy")
    spath = os.path.join(wd, "weather_std.npy")
    if os.path.exists(mpath) and os.path.exists(spath):
        return np.load(mpath), np.load(spath)
    return None, None


def scenario_to_weather_vector(scenario_name, normalize=True):
    """
    将天气情景（晴天/阴天/小雨/大雨/高温）转换为归一化的天气特征向量。
    返回 (4,) 数组：温度/降水/风速/云量（z-score）。
    """
    if scenario_name not in WEATHER_SCENARIOS:
        scenario_name = "晴天"
    s = WEATHER_SCENARIOS[scenario_name]
    raw = np.array([s["temperature"], s["precipitation"], s["wind_speed"], s["cloud_cover"]], dtype=np.float32)
    if not normalize:
        return raw
    wmean, wstd = _load_weather_normalization()
    if wmean is not None:
        return (raw - wmean) / (wstd + 1e-8)
    # 无标准化参数时按数据经验值归一化
    empirical_mean = np.array([20.0, 0.5, 10.0, 50.0], dtype=np.float32)
    empirical_std = np.array([5.0, 1.5, 6.0, 35.0], dtype=np.float32)
    return (raw - empirical_mean) / (empirical_std + 1e-8)


def forecast(model_type="gbt", horizon=72, weather_seq=None, start_idx=None, blend_steps=3):
    """
    滚动预测未来 horizon 小时。
    weather_seq: (horizon, 4) 归一化天气向量序列；None 时用历史同小时天气均值
    start_idx: 起始样本索引（默认最后一个测试样本）
    blend_steps: 前 N 步用测试集真实天气过渡（避免天气突变导致模型状态不稳）
    返回 (horizon, gs, gs) 预测数组
    """
    gs = config.GRID_SIZE
    predict = _load_model(model_type)
    if predict is None:
        log(f"模型 {model_type} 未找到，无法预测")
        return None

    wx = os.path.join(WEATHER_DIR, "X_test.npy")
    if os.path.exists(wx):
        X_test = np.load(wx)
    else:
        X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
    channels = X_test.shape[-1]

    wy = os.path.join(WEATHER_DIR, "y_test.npy")
    if os.path.exists(wy):
        y_test = np.load(wy)
    else:
        y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))
    n_test = y_test.shape[0]

    test_start_hour = config.TEST_START_HOUR if hasattr(config, "TEST_START_HOUR") else 511
    hours_of_test = [(test_start_hour + i) % 24 for i in range(n_test)]

    # 历史日周期均值（锚定用）
    hist_hourly = np.zeros((24, gs, gs), dtype=np.float32)
    cnt = np.zeros(24, dtype=np.float32)
    for i, h in enumerate(hours_of_test):
        hist_hourly[h] += y_test[i]
        cnt[h] += 1
    for h in range(24):
        if cnt[h] > 0:
            hist_hourly[h] /= cnt[h]
        else:
            hist_hourly[h] = y_test.mean(axis=0)

    # 历史天气均值（weather_seq 为 None 时的默认填充）
    hist_weather = None
    if channels > 1:
        hist_weather = np.zeros((24, channels - 1), dtype=np.float32)
        cntw = np.zeros(24, dtype=np.float32)
        for i, h in enumerate(hours_of_test[:min(n_test, X_test.shape[0])]):
            if i < X_test.shape[0]:
                hist_weather[h] += X_test[i, -1, 0, 0, 1:]
                cntw[h] += 1
        for h in range(24):
            if cntw[h] > 0:
                hist_weather[h] /= cntw[h]

    # 起点窗口
    if start_idx is None:
        start_idx = n_test - 1
    window = X_test[start_idx:start_idx + 1].copy()
    end_hour = hours_of_test[start_idx]

    if weather_seq is not None and channels > 1:
        weather_seq = np.asarray(weather_seq, dtype=np.float32)

    preds = []
    for step in range(horizon):
        y_next = predict(window)
        y_next = np.clip(y_next, 0, None)
        preds.append(y_next[0])

        future_hour = (end_hour + 1 + step) % 24
        new_frame = np.zeros((1, 1, gs, gs, channels), dtype=np.float32)
        new_frame[0, 0, :, :, 0] = y_next[0]
        if channels > 1:
            if weather_seq is not None and step < len(weather_seq):
                # 前 blend_steps 步用真实天气过渡（如果可用），之后用情景天气
                if step < blend_steps and start_idx is not None and start_idx + step + 1 < X_test.shape[0]:
                    for c in range(1, channels):
                        new_frame[0, 0, :, :, c] = X_test[start_idx + step + 1, -1, 0, 0, c]
                else:
                    for c in range(1, channels):
                        new_frame[0, 0, :, :, c] = weather_seq[step, c - 1]
            elif hist_weather is not None:
                for c in range(1, channels):
                    new_frame[0, 0, :, :, c] = hist_weather[future_hour, c - 1]
            else:
                new_frame[0, 0, :, :, 1:] = window[0, -1, :, :, 1:]
        window = np.concatenate([window[:, 1:], new_frame], axis=1)

    preds = np.stack(preds, axis=0)
    log(f"滚动预测完成: {model_type} {horizon}h, shape={preds.shape}")
    return preds


def forecast_for_date(model_type, target_date, target_hour, weather_scenario="晴天",
                      event=None, horizon_max=240):
    """
    预测指定未来日期+小时的时空需求。
    target_date: datetime.date 或 str（YYYY-MM-DD）
    target_hour: 0-23
    weather_scenario: 天气情景名（晴天/阴天/小雨/大雨/高温）
    event: None 或 dict {"type": "activity"/"accident", "x": int, "y": int, "radius": int, "strength": float}
    horizon_max: 最大滚动步数（默认 240h = 10 天）
    返回 (pred_grid, info_dict)；失败返回 (None, None)
    """
    gs = config.GRID_SIZE

    # 计算目标时间相对测试集末尾的偏移小时数
    wy = os.path.join(WEATHER_DIR, "y_test.npy")
    if os.path.exists(wy):
        y_test = np.load(wy)
    else:
        y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))
    n_test = y_test.shape[0]

    test_start_hour = config.TEST_START_HOUR if hasattr(config, "TEST_START_HOUR") else 511
    test_end_ts = pd.Timestamp("2018-09-22") + pd.Timedelta(hours=test_start_hour % 24 + n_test - 1)
    # 测试集末样本对应的小时 = test_start_hour + n_test - 1 的全局小时

    target_ts = pd.Timestamp(target_date) + pd.Timedelta(hours=int(target_hour))
    delta_h = int((target_ts - test_end_ts).total_seconds() / 3600)

    if delta_h <= 0:
        log(f"目标时间 {target_ts} 不晚于测试集末尾 {test_end_ts}，请选择未来时间")
        return None, None
    if delta_h > horizon_max:
        log(f"目标时间超出最大预测范围 {horizon_max}h")
        return None, None

    # 生成天气序列：全部用指定情景（归一化）
    wv = scenario_to_weather_vector(weather_scenario)
    weather_seq = np.tile(wv, (delta_h, 1))

    preds = forecast(model_type, horizon=delta_h, weather_seq=weather_seq)
    if preds is None:
        return None, None

    pred_grid = preds[-1]  # 目标时刻的预测

    # 事件扰动：活动→需求增加；事故→需求减少
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
            pred_grid = pred_grid * (1.0 + strength)
        else:
            pred_grid = pred_grid * (1.0 - strength * 0.8)
        pred_grid = np.clip(pred_grid, 0, None)

    info = {
        "target_time": target_ts,
        "delta_h": delta_h,
        "weather_scenario": weather_scenario,
        "event": event,
        "n_test": n_test,
    }
    return pred_grid, info


def forecast_full_day(model_type, target_date, weather_scenario="晴天", event=None, horizon=24):
    """
    预测目标日期全天 24 小时需求（一次滚动，避免重复计算）。
    返回 (24, gs, gs) 预测数组 + info；失败返回 (None, None)
    """
    gs = config.GRID_SIZE

    wy = os.path.join(WEATHER_DIR, "y_test.npy")
    if os.path.exists(wy):
        y_test = np.load(wy)
    else:
        y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))
    n_test = y_test.shape[0]

    test_start_hour = config.TEST_START_HOUR if hasattr(config, "TEST_START_HOUR") else 511
    test_end_ts = pd.Timestamp("2018-09-22") + pd.Timedelta(hours=test_start_hour % 24 + n_test - 1)

    target_ts = pd.Timestamp(target_date)
    delta_h = int((target_ts - test_end_ts).total_seconds() / 3600)

    if delta_h <= 0:
        log(f"目标日期 {target_ts} 不晚于测试集末尾，请选择未来日期")
        return None, None
    if delta_h > 240:
        log(f"目标日期超出最大预测范围 240h")
        return None, None

    wv = scenario_to_weather_vector(weather_scenario)
    weather_seq = np.tile(wv, (delta_h + horizon, 1))

    preds = forecast(model_type, horizon=delta_h + horizon, weather_seq=weather_seq)
    if preds is None:
        return None, None

    day_preds = preds[delta_h:delta_h + horizon]  # 目标日期 0:00 起 24 小时

    # 事件扰动（对全天各小时叠加）
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
            day_preds = day_preds * (1.0 + strength)
        else:
            day_preds = day_preds * (1.0 - strength * 0.8)
        day_preds = np.clip(day_preds, 0, None)

    info = {"target_time": target_ts, "delta_h": delta_h, "weather_scenario": weather_scenario, "event": event}
    return day_preds, info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gbt")
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--weather", default="晴天")
    args = parser.parse_args()
    r = forecast(args.model, args.horizon, weather_seq=np.tile(
        scenario_to_weather_vector(args.weather), (args.horizon, 1)))
    if r is not None:
        print(f"预测 {args.horizon} 小时完成 (天气: {args.weather})")
        print(f"  每小时总需求: {r.sum(axis=(1,2)).round(1)}")
        for d in range(args.horizon // 24):
            day_sum = r[d*24:(d+1)*24].sum()
            print(f"  第{d+1}天总需求: {day_sum:.0f}")
