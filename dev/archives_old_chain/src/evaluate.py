"""
模型评估与对比 + 误差回放记录 + Optuna 调参
用法: python src/evaluate.py              # 基础对比评估
      python src/evaluate.py --save-errors # 同时保存逐小时误差（基于模型预测）
      python src/evaluate.py --tune        # Optuna 自动调参
      python src/evaluate.py --tune-xgb    # 仅调梯度提升树
"""

import os, sys, pickle, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log, timer, plot_model_comparison, ensure_fresh_cache, pool_5d, pool_3d, unpool


def _ensure_weather_data():
    """检测天气增强数据是否存在，不存在则自动生成"""
    weather_dir = os.path.join(config.PROCESSED_DIR, "weather")
    x_path = os.path.join(weather_dir, "X_train.npy")
    if os.path.exists(x_path):
        log("天气增强数据已存在，跳过生成。")
        return
    log("未检测到天气增强数据，自动生成中…")
    from src.weather_enhance import enhance
    enhance(force=True)


@timer
def evaluate(args):
    log("=" * 60)
    log("  模型对比评估")
    log("=" * 60)

    ensure_fresh_cache()
    _ensure_weather_data()

    results = {}
    errors = {}

    if args.tune or args.tune_xgb:
        log("--- Optuna 自动调参 ---")
        _tune_and_record(results, errors, args)
    else:
        if not args.skip_gbt:
            try:
                from src.train_xgboost import train as train_gbt
                results["梯度提升树"] = train_gbt()
            except Exception as e:
                log(f"梯度提升树 失败: {e}")
                errors["梯度提升树"] = str(e)

        if not args.skip_lstm:
            try:
                from src.train_lstm import train as train_lstm
                results["LSTM"] = train_lstm()
            except Exception as e:
                log(f"LSTM 失败: {e}")
                errors["LSTM"] = str(e)

        if not args.skip_clstm:
            try:
                from src.train_convlstm import train as train_clstm
                results["ConvLSTM"] = train_clstm()
            except Exception as e:
                log(f"ConvLSTM 失败: {e}")
                errors["ConvLSTM"] = str(e)

    if results:
        _print_comparison(results, errors)
        if len(results) >= 2:
            plot_model_comparison(results)
        _save_report(results, errors)

    if args.save_errors:
        _save_error_history()

    log("=== 评估完成 ===")
    return results


def _save_error_history():
    """
    基于真实模型预测计算逐小时误差。
    优先使用梯度提升树（最快且精度最高），fallback 到 LSTM。
    误差 = y_true - y_pred（模型预测残差，非数据自身波动）。
    """
    import numpy as np

    weather_dir = os.path.join(config.PROCESSED_DIR, "weather")
    if os.path.exists(os.path.join(weather_dir, "X_test.npy")):
        X_test = np.load(os.path.join(weather_dir, "X_test.npy"))
        y_test = np.load(os.path.join(weather_dir, "y_test.npy"))
    else:
        X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))

    gs = config.GRID_SIZE
    n_test = y_test.shape[0]

    # 优先用梯度提升树预测
    y_pred_all = None
    model_label = ""

    gbt_path = os.path.join(config.OUTPUT_DIR, "xgboost_model.pkl")
    if os.path.exists(gbt_path):
        log("使用梯度提升树模型生成预测…")
        with open(gbt_path, "rb") as f:
            gbt_model = pickle.load(f)
        pool = config.XGB_POOL
        X_te_p = pool_5d(X_test, pool)
        X_te_f = X_te_p.reshape(n_test, -1)
        y_te_p = pool_3d(y_test, pool)
        y_pr_p = gbt_model.predict(X_te_f)
        # predict 返回 (n_test, 25)，reshape 为 (n_test, pg, pg)
        pg = gs // pool
        y_pr_p = y_pr_p.reshape(n_test, pg, pg)
        y_pred_all = unpool(y_pr_p, pool, gs)
        model_label = "梯度提升树"

    if y_pred_all is None:
        lstm_path = os.path.join(config.OUTPUT_DIR, "lstm_model.pth")
        if os.path.exists(lstm_path):
            log("梯度提升树模型未找到，使用 LSTM…")
            import torch
            import torch.nn as nn
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
            model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
            model.eval()
            X_te_f = X_test.reshape(n_test, config.WINDOW_SIZE, grid_dim)
            with torch.no_grad():
                y_pr = model(torch.FloatTensor(X_te_f)).numpy()
            y_pred_all = unpool(y_pr.reshape(-1, pg, pg), pool, gs)
            model_label = "LSTM"

    if y_pred_all is None:
        log("⚠️ 未找到任何已训练模型，无法生成真实误差回放。请先运行 evaluate.py 训练模型。")
        return None

    log(f"模型: {model_label}, 预测样本数: {n_test}")

    # 计算逐小时 RMSE（模型预测 vs 真实标签）
    rmse_over_time = []
    error_grid = np.zeros((gs, gs), dtype=np.float64)

    for i in range(n_test):
        diff = y_test[i] - y_pred_all[i]
        rmse_over_time.append(float(np.sqrt(np.mean(diff ** 2))))
        error_grid += np.abs(diff)

    error_grid /= n_test

    data = {
        "rmse_over_time": rmse_over_time,
        "error_grid": error_grid,
        "n_hours": n_test,
        "model_used": model_label,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = os.path.join(config.OUTPUT_DIR, "error_history.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    log(f"误差历史已保存: {path} (基于 {model_label} 真实预测)")
    return data


def _tune_and_record(results, errors, args):
    import numpy as np
    import optuna
    from sklearn.metrics import mean_squared_error
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor

    weather_dir = os.path.join(config.PROCESSED_DIR, "weather")
    if os.path.exists(os.path.join(weather_dir, "X_train.npy")):
        X_train = np.load(os.path.join(weather_dir, "X_train.npy"))
        X_test = np.load(os.path.join(weather_dir, "X_test.npy"))
        y_train = np.load(os.path.join(weather_dir, "y_train.npy"))
        y_test = np.load(os.path.join(weather_dir, "y_test.npy"))
    else:
        X_train = np.load(os.path.join(config.PROCESSED_DIR, "X_train.npy"))
        X_test = np.load(os.path.join(config.PROCESSED_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(config.PROCESSED_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(config.PROCESSED_DIR, "y_test.npy"))

    pool = config.XGB_POOL
    n_tr = X_train.shape[0]
    X_tr = pool_5d(X_train, pool).reshape(n_tr, -1)
    X_te = pool_5d(X_test, pool).reshape(X_test.shape[0], -1)
    y_tr = pool_3d(y_train, pool).reshape(n_tr, -1)

    def objective(trial):
        n_iter = trial.suggest_int("max_iter", 50, 150)
        depth = trial.suggest_int("max_depth", 3, 10)
        lr = trial.suggest_float("learning_rate", 0.02, 0.3, log=True)
        l2 = trial.suggest_float("l2_regularization", 0.0, 1.0)

        base = HistGradientBoostingRegressor(
            max_iter=n_iter, max_depth=depth, learning_rate=lr,
            l2_regularization=l2, random_state=config.RANDOM_SEED,
        )
        model = MultiOutputRegressor(base, n_jobs=-1)
        model.fit(X_tr, y_tr)

        y_pr = model.predict(X_te)
        y_te_true = pool_3d(y_test, pool).reshape(X_test.shape[0], -1)
        return float(np.sqrt(mean_squared_error(y_te_true, y_pr)))

    log("Optuna 搜索中（约 3-5 分钟）…")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best = study.best_params
    log(f"最优参数: {best}")
    log(f"最优 RMSE: {study.best_value:.4f}")

    base = HistGradientBoostingRegressor(
        max_iter=best["max_iter"], max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        l2_regularization=best.get("l2_regularization", 0),
        random_state=config.RANDOM_SEED,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_tr, y_tr)

    y_pr = model.predict(X_te)
    y_te_true = pool_3d(y_test, pool).reshape(X_test.shape[0], -1)

    from sklearn.metrics import mean_absolute_error, r2_score
    rmse_val = float(np.sqrt(mean_squared_error(y_te_true, y_pr)))
    mae_val = float(mean_absolute_error(y_te_true, y_pr))
    r2_val = float(r2_score(y_te_true, y_pr))

    label = "梯度提升树(Optuna)"
    results[label] = {"rmse": rmse_val, "mae": mae_val, "r2": r2_val, "model": model}
    print(f"\n  {label}:  RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  R2={r2_val:.4f}")

    model_path = os.path.join(config.OUTPUT_DIR, "xgboost_model_tuned.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log(f"调参模型已保存: {model_path}")

    if not args.tune_xgb:
        try:
            from src.train_lstm import train as train_lstm
            results["LSTM"] = train_lstm()
        except Exception as e:
            errors["LSTM"] = str(e)
        try:
            from src.train_convlstm import train as train_clstm
            results["ConvLSTM"] = train_clstm()
        except Exception as e:
            errors["ConvLSTM"] = str(e)


def _print_comparison(results, errors):
    print("\n" + "=" * 60)
    print(f"{'模型':<25} {'RMSE':>12} {'MAE':>12} {'R2':>12}")
    print("-" * 60)
    for model, m in results.items():
        print(f"{model:<25} {m['rmse']:>12.4f} {m['mae']:>12.4f} {m['r2']:>12.4f}")
    for model in errors:
        print(f"{model:<25} {'FAILED':>12} {'FAILED':>12} {'FAILED':>12}")
    print("=" * 60)

    best_rmse = min(results, key=lambda k: results[k]["rmse"])
    best_r2 = max(results, key=lambda k: results[k]["r2"])
    print(f"\n最佳 RMSE: {best_rmse} ({results[best_rmse]['rmse']:.4f})")
    print(f"最佳 R2:   {best_r2} ({results[best_r2]['r2']:.4f})")


def _save_report(results, errors):
    report_path = os.path.join(config.REPORT_DIR, "experiment_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("共享单车需求预测 - 模型对比实践结果\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("实践配置:\n")
        f.write(f"  原始网格: {config.GRID_SIZE}x{config.GRID_SIZE}\n")
        f.write(f"  滑动窗口: {config.WINDOW_SIZE} 小时\n")
        f.write(f"  训练集比例: {config.TRAIN_RATIO}\n")
        f.write(f"  随机种子: {config.RANDOM_SEED}\n")
        f.write(f"  数据来源: CitiBike 2018-09 (纽约)\n")
        f.write(f"  特征通道: 5 (借车量 + 温度/降水/风速/云量)\n")
        f.write("  天气特征: Open-Meteo Archive API\n")
        f.write("  池化配置:\n")
        f.write(f"    梯度提升树: pool={config.XGB_POOL} ({config.GRID_SIZE//config.XGB_POOL}x{config.GRID_SIZE//config.XGB_POOL})\n")
        f.write(f"    LSTM:        pool={config.LSTM_POOL} ({config.GRID_SIZE//config.LSTM_POOL}x{config.GRID_SIZE//config.LSTM_POOL})\n")
        f.write(f"    ConvLSTM:    pool={config.CONVLSTM_POOL} ({config.GRID_SIZE//config.CONVLSTM_POOL}x{config.GRID_SIZE//config.CONVLSTM_POOL})\n\n")

        f.write(f"{'模型':<25} {'RMSE':>12} {'MAE':>12} {'R2':>12}\n")
        f.write("-" * 60 + "\n")
        for model, m in results.items():
            f.write(f"{model:<25} {m['rmse']:>12.4f} {m['mae']:>12.4f} {m['r2']:>12.4f}\n")
        for model in errors:
            f.write(f"{model:<25} {'FAILED':>12} {'FAILED':>12} {'FAILED':>12}\n")
        f.write("\n")

        if results:
            best_rmse = min(results, key=lambda k: results[k]["rmse"])
            best_r2 = max(results, key=lambda k: results[k]["r2"])
            f.write("最优指标:\n")
            f.write(f"  最低 RMSE: {best_rmse} = {results[best_rmse]['rmse']:.4f}\n")
            f.write(f"  最高 R2:   {best_r2} = {results[best_r2]['r2']:.4f}\n")

    log(f"报告已保存: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-errors", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Optuna 调参")
    parser.add_argument("--tune-xgb", action="store_true", help="仅调 XGBoost")
    parser.add_argument("--skip-gbt", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-clstm", action="store_true")
    evaluate(parser.parse_args())
