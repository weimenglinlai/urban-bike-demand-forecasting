"""
LSTM 模型训练与评估（基于 PyTorch）
策略：空间池化降维 y，自动检测天气增强数据（5通道）
"""

import os, sys, pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log, timer, plot_loss_curve, plot_scatter, pool_3d

MODEL_FILENAME = "lstm_model.pth"
WEATHER_DIR = os.path.join(config.PROCESSED_DIR, "weather")


def _load_data():
    x_path = os.path.join(WEATHER_DIR, "X_train.npy")
    if os.path.exists(x_path):
        log("使用天气增强数据 (使用全部通道)")
        return (
            np.load(x_path),
            np.load(os.path.join(WEATHER_DIR, "X_test.npy")),
            np.load(os.path.join(WEATHER_DIR, "y_train.npy")),
            np.load(os.path.join(WEATHER_DIR, "y_test.npy")),
        )
    return (
        np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy")),
        np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy")),
        np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy")),
        np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy")),
    )


@timer
def train():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    log(f"=== LSTM 训练开始（设备: {device}, GPU: {gpu_name}） ===")

    X_train, X_test, y_train, y_test = _load_data()

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    gs = config.GRID_SIZE
    pool = config.LSTM_POOL
    pg = gs // pool
    channels = X_train.shape[-1]

    y_tr_p = pool_3d(y_train, pool)
    y_te_p = pool_3d(y_test, pool)
    y_tr_f = y_tr_p.reshape(n_train, -1)
    y_te_f = y_te_p.reshape(n_test, -1)

    grid_dim = gs * gs * channels
    out_dim = pg * pg

    X_train_f = X_train.reshape(n_train, config.WINDOW_SIZE, grid_dim)
    X_test_f = X_test.reshape(n_test, config.WINDOW_SIZE, grid_dim)

    log(f"池化 {gs}→{pg}, channels={channels}, grid_dim={grid_dim}, out={out_dim}")

    p = config.LSTM_PARAMS

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(grid_dim, p["hidden_size"], p["num_layers"],
                                batch_first=True,
                                dropout=p["dropout"] if p["num_layers"] > 1 else 0)
            self.fc = nn.Linear(p["hidden_size"], out_dim)

        def forward(self, x):
            h0 = torch.zeros(p["num_layers"], x.size(0), p["hidden_size"], device=x.device)
            c0 = torch.zeros(p["num_layers"], x.size(0), p["hidden_size"], device=x.device)
            out, _ = self.lstm(x, (h0, c0))
            return self.fc(out[:, -1, :])

    model = LSTMModel().to(device)

    val_split = int(n_train * 0.9)
    X_tr_t = torch.FloatTensor(X_train_f[:val_split])
    y_tr_t = torch.FloatTensor(y_tr_f[:val_split])
    X_val_t = torch.FloatTensor(X_train_f[val_split:])
    y_val_t = torch.FloatTensor(y_tr_f[val_split:])

    train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=p["batch_size"], shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=p["batch_size"])

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=p["lr_reduce_factor"],
        patience=p["lr_reduce_patience"])
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(p["epochs"]):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        train_losses.append(epoch_loss / len(train_dl.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
        val_losses.append(val_loss / len(val_dl.dataset))
        scheduler.step(val_losses[-1])

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  Epoch {epoch+1:3d}/{p['epochs']} | "
                f"Train: {train_losses[-1]:.2f} | Val: {val_losses[-1]:.2f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.1e}")

        if patience_counter >= p["early_stop_patience"]:
            log(f"早停 Epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    log(f"最佳 Val Loss: {best_val_loss:.2f}")

    model.eval()
    X_te_t = torch.FloatTensor(X_test_f).to(device)
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy()

    rmse_val = float(np.sqrt(mean_squared_error(y_te_f, y_pred)))
    mae_val = float(mean_absolute_error(y_te_f, y_pred))
    r2_val = float(r2_score(y_te_f, y_pred))

    print(f"\n  LSTM（{pg}×{pg} 池化, {channels}通道）:")
    print(f"    RMSE: {rmse_val:.4f}  MAE: {mae_val:.4f}  R2: {r2_val:.4f}")

    model_path = os.path.join(config.OUTPUT_DIR, MODEL_FILENAME)
    torch.save(model.state_dict(), model_path)
    log(f"模型已保存: {model_path}")

    plot_loss_curve(train_losses, val_losses, "LSTM 训练曲线", "lstm_loss_curve.png")
    plot_scatter(y_te_f, y_pred, f"LSTM（{pg}×{pg}）预测值 vs 真实值", "lstm_scatter.png")
    log("=== LSTM 训练完成 ===")
    return {"rmse": rmse_val, "mae": mae_val, "r2": r2_val, "model": model}


if __name__ == "__main__":
    train()
