"""
ConvLSTM 模型训练与评估（基于 PyTorch）
策略：空间池化 X 和 y，降低卷积核在高分辨率下的无效开销
"""

import os, sys, time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log, timer, plot_loss_curve, plot_scatter, pool_5d, pool_3d

MODEL_FILENAME = "convlstm_model.pth"
WEATHER_DIR = os.path.join(config.PROCESSED_DIR, "weather")


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad_h, pad_w = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=(pad_h, pad_w), bias=bias)

    def forward(self, x, h_cur):
        h, c = h_cur
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(self, batch_size, height, width):
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
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


@timer
def train():
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    log(f"=== ConvLSTM 训练开始（设备: {device}, GPU: {gpu_name}） ===")

    x_path = os.path.join(WEATHER_DIR, "X_train.npy")
    if os.path.exists(x_path):
        log("使用天气增强数据")
        X_train = np.load(x_path)
        X_test = np.load(os.path.join(WEATHER_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(WEATHER_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(WEATHER_DIR, "y_test.npy"))
    else:
        X_train = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))
        X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))

    pool = config.CONVLSTM_POOL
    gs = config.GRID_SIZE
    pg = gs // pool

    X_tr_p = pool_5d(X_train, pool)
    X_te_p = pool_5d(X_test, pool)
    y_tr_p = pool_3d(y_train, pool)
    y_te_p = pool_3d(y_test, pool)

    n_train, n_test = X_tr_p.shape[0], X_te_p.shape[0]

    X_train_t = X_tr_p.astype(np.float32)
    X_test_t = X_te_p.astype(np.float32)
    y_train_t = y_tr_p.astype(np.float32)[:, np.newaxis, :, :]
    y_test_t = y_te_p.astype(np.float32)[:, np.newaxis, :, :]

    log(f"池化 {gs}→{pg}, X={X_train_t.shape}, y={y_train_t.shape}")

    p = config.CONVLSTM_PARAMS
    model = ConvLSTM(
        in_channels=X_train_t.shape[-1],
        hidden_channels_list=p["hidden_channels"],
        kernel_size=p["kernel_size"],
        output_ch=1,
    ).to(device)

    val_split = int(n_train * 0.9)
    X_tr = torch.FloatTensor(X_train_t[:val_split])
    y_tr = torch.FloatTensor(y_train_t[:val_split])
    X_val = torch.FloatTensor(X_train_t[val_split:])
    y_val = torch.FloatTensor(y_train_t[val_split:])

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=p["batch_size"], shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=p["batch_size"])

    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    t_start = time.time()

    for epoch in range(p["epochs"]):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_dl:
            bx = bx.to(device).permute(0, 1, 4, 2, 3)
            by = by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        train_losses.append(epoch_loss / len(train_dl.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_dl:
                bx = bx.to(device).permute(0, 1, 4, 2, 3)
                by = by.to(device)
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

        if (time.time() - t_start) / 60 > p["max_training_minutes"]:
            log(f"超时 {p['max_training_minutes']}min")
            break

    if best_state:
        model.load_state_dict(best_state)
    log(f"最佳 Val Loss: {best_val_loss:.2f}")

    model.eval()
    X_te_pt = torch.FloatTensor(X_test_t).to(device).permute(0, 1, 4, 2, 3)
    with torch.no_grad():
        y_pred = model(X_te_pt).cpu().numpy()

    y_pred_f = y_pred.reshape(-1, pg * pg)
    y_test_f = y_test_t.reshape(-1, pg * pg)

    rmse_val = float(np.sqrt(mean_squared_error(y_test_f, y_pred_f)))
    mae_val = float(mean_absolute_error(y_test_f, y_pred_f))
    r2_val = float(r2_score(y_test_f, y_pred_f))

    print(f"\n  ConvLSTM（{pg}×{pg} 池化）:")
    print(f"    RMSE: {rmse_val:.4f}  MAE: {mae_val:.4f}  R2: {r2_val:.4f}")

    model_path = os.path.join(config.OUTPUT_DIR, MODEL_FILENAME)
    torch.save(model.state_dict(), model_path)
    log(f"模型已保存: {model_path}")

    plot_loss_curve(train_losses, val_losses, "ConvLSTM 训练曲线", "convlstm_loss_curve.png")
    plot_scatter(y_test_f, y_pred_f, f"ConvLSTM（{pg}×{pg}）预测值 vs 真实值", "convlstm_scatter.png")
    log("=== ConvLSTM 训练完成 ===")
    return {"rmse": rmse_val, "mae": mae_val, "r2": r2_val, "model": model}


if __name__ == "__main__":
    train()
