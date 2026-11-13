"""
梯度提升树基线模型训练与评估
策略：空间池化降维，sklearn HistGradientBoostingRegressor
自动检测 weather 增强数据（5 通道），不存在则 fallback 原始 1 通道。
参数全部从 config.XGB_PARAMS 读取。
"""

import os, sys, time, pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log, timer, plot_scatter, pool_5d, pool_3d

MODEL_FILENAME = "xgboost_model.pkl"

WEATHER_PROCESSED = os.path.join(config.PROCESSED_DIR, "weather")


def _load_data():
    """加载数据，优先天气增强版"""
    x_path = os.path.join(WEATHER_PROCESSED, "X_train.npy")
    if os.path.exists(x_path):
        log("使用天气增强数据 (5通道)")
        return (
            np.load(x_path),
            np.load(os.path.join(WEATHER_PROCESSED, "X_test.npy")),
            np.load(os.path.join(WEATHER_PROCESSED, "y_train.npy")),
            np.load(os.path.join(WEATHER_PROCESSED, "y_test.npy")),
        )
    return (
        np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy")),
        np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy")),
        np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy")),
        np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy")),
    )


@timer
def train():
    log("=== 梯度提升树基线训练 ===")

    X_train, X_test, y_train, y_test = _load_data()
    pool = config.XGB_POOL
    n_tr, n_te = X_train.shape[0], X_test.shape[0]

    X_tr_p = pool_5d(X_train, pool)
    X_te_p = pool_5d(X_test, pool)
    y_tr_p = pool_3d(y_train, pool)
    y_te_p = pool_3d(y_test, pool)

    Hp = X_tr_p.shape[2]
    dim_out = Hp * Hp

    X_tr_f = X_tr_p.reshape(n_tr, -1)
    X_te_f = X_te_p.reshape(n_te, -1)
    y_tr_f = y_tr_p.reshape(n_tr, dim_out)
    y_te_f = y_te_p.reshape(n_te, dim_out)

    log(f"池化 {config.GRID_SIZE}→{Hp}, 特征={X_tr_f.shape[1]}, 输出={dim_out}")

    params = dict(config.XGB_PARAMS)
    log(f"参数: max_iter={params['max_iter']}, max_depth={params['max_depth']}, "
        f"learning_rate={params['learning_rate']}")

    t0 = time.time()
    base = HistGradientBoostingRegressor(**params)
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_tr_f, y_tr_f)
    log(f"训练 {time.time() - t0:.1f}s")

    t0 = time.time()
    y_pr = model.predict(X_te_f)
    log(f"预测 {time.time() - t0:.1f}s")

    rmse_val = float(np.sqrt(mean_squared_error(y_te_f, y_pr)))
    mae_val = float(mean_absolute_error(y_te_f, y_pr))
    r2_val = float(r2_score(y_te_f, y_pr))

    print(f"\n  梯度提升树（{Hp}×{Hp} 池化）:")
    print(f"    RMSE: {rmse_val:.4f}  MAE: {mae_val:.4f}  R2: {r2_val:.4f}")

    model_path = os.path.join(config.OUTPUT_DIR, MODEL_FILENAME)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log(f"模型已保存: {model_path}")

    plot_scatter(y_te_f, y_pr, f"梯度提升树（{Hp}×{Hp}）预测值 vs 真实值", "xgboost_scatter.png")
    log("=== 梯度提升树训练完成 ===")
    return {"rmse": rmse_val, "mae": mae_val, "r2": r2_val, "model": model}


if __name__ == "__main__":
    train()
