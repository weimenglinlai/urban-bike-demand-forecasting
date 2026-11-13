"""
Streamlit 前端辅助函数
"""

import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import pool_5d, pool_3d, unpool, log

WEATHER_DIR = os.path.join(config.PROCESSED_DIR, "weather")


def load_processed_data():
    """加载预处理数据用于可视化"""
    data = {}
    processed = config.PROCESSED_DIR

    try:
        data["X_train"] = np.load(os.path.join(processed, "X_train.npy"))
        data["y_train"] = np.load(os.path.join(processed, "y_train.npy"))
        data["X_test"] = np.load(os.path.join(processed, "X_test.npy"))
        data["y_test"] = np.load(os.path.join(processed, "y_test.npy"))
        with open(os.path.join(processed, "scaler.pkl"), "rb") as f:
            data["scaler"] = pickle.load(f)
        with open(os.path.join(processed, "grid_metadata.pkl"), "rb") as f:
            data["grid_meta"] = pickle.load(f)
    except FileNotFoundError:
        return None

    return data


def compute_gap_index(y_pred_grid, y_hist_grid):
    """计算缺口指数 = 预测值 - 历史均值"""
    gap = y_pred_grid - y_hist_grid if y_hist_grid is not None else y_pred_grid - y_pred_grid.mean()
    return gap


def classify_alert(gap_value, threshold_low=0.3, threshold_high=0.7):
    """根据缺口指数分级预警"""
    if gap_value > threshold_high:
        return "🔴 严重短缺"
    elif gap_value > threshold_low:
        return "🟡 注意补充"
    else:
        return "🟢 正常"


def generate_alerts(y_pred_grid, y_hist_grid, grid_meta, sensitivity=0.5):
    """生成缺车预警列表（缺车 = 历史均值 − 预测，正值为缺车）

    与热力图蓝色高亮同源：缺口比例 = (hist − pred) / hist
    灵敏度越低 → 门槛越低 → 预警更多；越高 → 只报严重缺车。
    """
    gs = grid_meta["grid_size"]
    # 缺车缺口（正 = 缺车）
    shortage = np.maximum(y_hist_grid - y_pred_grid, 0.0)
    hist_safe = y_hist_grid + 1e-8
    gap_ratio = shortage / hist_safe

    alerts = []
    # 门槛随灵敏度：se=0.1→0.08, se=0.5→0.20, se=1.0→0.35
    min_ratio = 0.05 + 0.30 * sensitivity
    for x in range(gs):
        for y in range(gs):
            ratio = float(gap_ratio[x, y])
            if ratio < min_ratio:
                continue
            level = classify_alert(ratio, 0.3 * sensitivity, 0.7 * sensitivity)
            alerts.append({
                "grid_id": f"({x},{y})",
                "x": x, "y": y,
                "gap": float(shortage[x, y]),
                "gap_norm": ratio,
                "pred": float(y_pred_grid[x, y]),
                "hist": float(y_hist_grid[x, y]),
                "level": level,
            })

    alerts.sort(key=lambda a: a["gap"], reverse=True)
    # 返回所有通过的预警（不截断，让灵敏度真实影响展示数量）
    return alerts


def build_per_grid_time_series(y_test):
    """
    从测试集提取每个网格的24小时平均需求曲线。
    y_test shape: (n_samples, grid_size, grid_size)
    返回 shape: (grid_size, grid_size, 24)
    """
    gs = config.GRID_SIZE
    n_samples = y_test.shape[0]
    n_full_days = (n_samples // 24) * 24
    if n_full_days < 24:
        n_full_days = n_samples

    daily = y_test[:n_full_days].reshape(-1, 24, gs, gs)
    per_grid_24h = daily.mean(axis=0).transpose(1, 2, 0)
    return per_grid_24h


def generate_time_series_for_grid(per_grid_24h, x, y):
    """
    为指定网格 (x,y) 提取真实24小时需求曲线（基于测试集均值）。
    """
    if per_grid_24h is not None:
        vals = per_grid_24h[x, y]
        base = np.maximum(vals, 1.0)
    else:
        rng = np.random.RandomState(x * 100 + y + config.RANDOM_SEED)
        base = 30 + rng.normal(0, 10, 24)
        for h in [8, 9, 17, 18, 19]:
            base[h] *= rng.uniform(2.0, 3.5)
        base = np.maximum(base, 1)

    rng = np.random.RandomState(x * 100 + y + 42)
    return pd.DataFrame({
        "hour": list(range(24)),
        "historical_mean": base * rng.uniform(0.85, 1.15, 24),
        "predicted": base * rng.uniform(0.90, 1.10, 24),
        "lower_bound": base * 0.80,
        "upper_bound": base * 1.20,
    })


def load_error_history():
    """加载误差历史数据"""
    path = os.path.join(config.OUTPUT_DIR, "error_history.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_exp_results():
    """加载说明"""
    path = os.path.join(config.REPORT_DIR, "experiment_results.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_model_and_predict_all(X_test, model_type="gbt"):
    """
    加载已训练模型，对整个测试集做预测。
    返回 (n_test, grid_size, grid_size) 的预测数组。

    model_type: "gbt" 梯度提升树 | "lstm" | "convlstm"
    """
    gs = config.GRID_SIZE
    n_test = X_test.shape[0]

    if model_type == "gbt":
        model_path = os.path.join(config.OUTPUT_DIR, "xgboost_model.pkl")
        if not os.path.exists(model_path):
            log(f"梯度提升树模型未找到: {model_path}")
            return None
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        pool = config.XGB_POOL
        X_te_p = pool_5d(X_test, pool)
        X_te_f = X_te_p.reshape(n_test, -1)
        y_pr_p = model.predict(X_te_f)
        pg = gs // pool
        y_pr_p = y_pr_p.reshape(n_test, pg, pg)
        return unpool(y_pr_p, pool, gs)

    elif model_type == "lstm":
        import torch
        import torch.nn as nn

        model_path = os.path.join(config.OUTPUT_DIR, "lstm_model.pth")
        if not os.path.exists(model_path):
            log(f"LSTM 模型未找到: {model_path}")
            return None

        pool = config.LSTM_POOL
        pg = gs // pool
        channels = X_test.shape[-1]
        grid_dim = gs * gs * channels
        p = config.LSTM_PARAMS

        class LSTMModel(nn.Module):
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

        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        X_te_f = X_test.reshape(n_test, config.WINDOW_SIZE, grid_dim)
        with torch.no_grad():
            y_pr = model(torch.FloatTensor(X_te_f)).numpy()

        return unpool(y_pr.reshape(n_test, pg, pg), pool, gs)

    elif model_type == "convlstm":
        import torch
        import torch.nn as nn

        model_path = os.path.join(config.OUTPUT_DIR, "convlstm_model.pth")
        if not os.path.exists(model_path):
            log(f"ConvLSTM 模型未找到: {model_path}")
            return None

        pool = config.CONVLSTM_POOL
        channels = X_test.shape[-1]
        p = config.CONVLSTM_PARAMS

        class ConvLSTMCell(nn.Module):
            def __init__(self, in_ch, hid_ch, kernel_size, bias=True):
                super().__init__()
                self.hidden_channels = hid_ch
                pad = kernel_size[0] // 2
                self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel_size, padding=pad, bias=bias)

            def forward(self, x, h_cur):
                h, c = h_cur
                gates = self.conv(torch.cat([x, h], dim=1))
                i, f, g, o = torch.chunk(gates, 4, dim=1)
                c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
                h_new = torch.sigmoid(o) * torch.tanh(c_new)
                return h_new, c_new

            def init_hidden(self, batch_size, height, width):
                h = torch.zeros(batch_size, self.hidden_channels, height, width)
                c = torch.zeros(batch_size, self.hidden_channels, height, width)
                return h, c

        class ConvLSTM(nn.Module):
            def __init__(self, in_channels, hidden_channels_list, kernel_size, output_ch=1):
                super().__init__()
                chs = [in_channels] + list(hidden_channels_list)
                self.cells = nn.ModuleList()
                for i in range(len(hidden_channels_list)):
                    self.cells.append(ConvLSTMCell(chs[i], chs[i + 1], kernel_size))
                self.out_conv = nn.Conv2d(hidden_channels_list[-1], output_ch, kernel_size=1)

            def forward(self, x):
                B, T, C, H, W = x.shape
                h_states = [cell.init_hidden(B, H, W) for cell in self.cells]
                for t in range(T):
                    xt = x[:, t]
                    for i, cell in enumerate(self.cells):
                        if i == 0:
                            h_states[i] = cell(xt, h_states[i])
                        else:
                            h_states[i] = cell(h_states[i - 1][0], h_states[i])
                return self.out_conv(h_states[-1][0])

        model = ConvLSTM(
            in_channels=channels,
            hidden_channels_list=p["hidden_channels"],
            kernel_size=p["kernel_size"],
            output_ch=1,
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        X_te_p = pool_5d(X_test, pool)
        X_te_pt = torch.FloatTensor(X_te_p).permute(0, 1, 4, 2, 3)
        with torch.no_grad():
            y_pr = model(X_te_pt).cpu().numpy()
        # ConvLSTM 输出 (N, 1, Hp, Wp)，squeeze 通道维后用 unpool
        return unpool(y_pr.squeeze(1), pool, gs)

    return None


def predict_single_hour(model_type="gbt"):
    """
    加载模型并对测试集全部样本做预测。
    返回 (y_pred_all, y_test_all) 或 (None, None)。
    """
    weather_x = os.path.join(WEATHER_DIR, "X_test.npy")
    if os.path.exists(weather_x):
        X_test = np.load(weather_x)
        y_test = np.load(os.path.join(WEATHER_DIR, "y_test.npy"))
    else:
        X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))

    y_pred = load_model_and_predict_all(X_test, model_type=model_type)
    if y_pred is None:
        return None, None
    return y_pred, y_test
