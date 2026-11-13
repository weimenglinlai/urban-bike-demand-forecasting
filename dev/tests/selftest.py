# -*- coding: utf-8 -*-
"""
端到端自测（不依赖 pytest）
============================
用法:  python tests/selftest.py

覆盖项
  1. 数据完整性       张量/时间戳/池化守恒/特征形状与 NaN
  2. 无数据泄漏       基线、气象归一化、标准化参数只由训练(核心)段估计；
                      人为置零测试段需求后，测试段基线必须完全不变
  3. 训练—推理一致    气象情景在推理时为**逐小时变化**（非全天常数）
  4. 同口径对照       四模型在同一测试小时集合上评估，样本数完全一致
  5. 冷启动信息对等   冷启动下序列模型的残差通道与掩码全为 0
  6. 指标可复现       由保存的预测重算 RMSE/MAE/R²，与记录值一致
  7. 模型有效性       均优于改造前同族模型；均不显著劣于条件基线；
                      序列机制在滚动口径下产生可量化增益
  8. 机制生效核查     ConvLSTM 输出空间非均匀、与 LSTM 不再逐值相同
  9. 预测接口         形状/非负/确定性/对天气、日期类型、节假日、事件敏感
 10. 应用冒烟         Streamlit AppTest 无异常
"""

import io
import os
import sys
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _find_project_root():
    """向上查找项目根（含 config.py 与 src/ 的目录），使本文件无论放在
    项目根下还是 _dev/tests/ 下都能正确定位。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if os.path.exists(os.path.join(cand, "config.py")) and \
           os.path.isdir(os.path.join(cand, "src")):
            return cand
    return os.path.dirname(os.path.dirname(here))


ROOT = _find_project_root()
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import config
from src import period_trainer as pt
from src.baseline import ConditionalBaseline
from src.date_features import (FEATURE_NAMES, make_features,
                              load_scenario_profiles, scenario_hourly_weather)
from src.metrics import core_metrics

# 改造前的同族模型历史结果（原项目 outputs 记录），用于「是否进步」的硬对照
LEGACY = {
    "gbt": {"rmse": 958.90, "r2": 0.9228},
    "lstm": {"rmse": 1091.42, "r2": 0.8987},
    "convlstm": {"rmse": 1223.29, "r2": 0.8728},
}

RESULTS, FAILED = [], []


def check(name, cond, detail=""):
    ok = bool(cond)
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  | {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)
    return ok


def section(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ------------------------------------------------------------------ 1
section("1. 数据完整性")
tensor, hours, gmeta = pt.build_hourly_demand()
hours = pd.to_datetime(hours)
check("需求张量形状", tensor.shape[1:] == (config.GRID_SIZE, config.GRID_SIZE),
      str(tensor.shape))
check("时间戳单调递增", bool(np.all(np.diff(hours.values) > np.timedelta64(0))),
      f"{len(hours)} 小时")
check("需求无 NaN / 无负值", (not np.isnan(tensor).any()) and bool((tensor >= 0).all()))

data = pt.prepare_data(tensor, hours, gmeta)
n, pg, pool = data["n"], data["pg"], data["pool"]
P = pg * pg
for k in ("z", "pooled", "B", "R"):
    check(f"{k} 无 NaN", not np.isnan(data[k]).any(), f"shape={data[k].shape}")
check("特征维度与 FEATURE_NAMES 一致", data["z"].shape[1] == len(FEATURE_NAMES),
      f"{data['z'].shape[1]} vs {len(FEATURE_NAMES)}")
check("求和池化总量守恒", np.allclose(data["pooled"].sum(), tensor.sum(), rtol=1e-4),
      f"{data['pooled'].sum():.0f} vs {tensor.sum():.0f}")

# ------------------------------------------------------------------ 2
section("2. 无数据泄漏核查")
core, val, test = data["idx_core"], data["idx_val"], data["idx_test"]
n_tr = data["n_tr"]

w_core = data["w_raw"][core]
check("气象归一化参数 = 核心段统计量",
      np.allclose(data["w_mean"], w_core.mean(axis=0), atol=1e-6))
check("气象归一化参数 ≠ 全量统计量（确认未用测试段归一化），已排除全量口径",
      not np.allclose(data["w_mean"], data["w_raw"].mean(axis=0), atol=1e-9))
check("目标/残差标准化量 = 核心段统计量",
      np.allclose(data["y_mu"], data["y"][core].mean(axis=0), atol=1e-5)
      and np.allclose(data["r_mu"], data["R"][core].reshape(-1, P).mean(axis=0), atol=1e-5))

# 基线可复现
bl2 = ConditionalBaseline(pg, config.BASELINE_K, config.BASELINE_LEVEL_WINDOW)
bl2.fit(data["pooled"][core], hours[core], data["dt_arr"][core], data["rb_arr"][core])
bl2.compute_rolling(data["pooled"], hours, data["dt_arr"], data["rb_arr"],
                    freeze_idx=n_tr, deploy_idx=n)
b2 = np.asarray(bl2.rolling, dtype=np.float32).copy()
b2[n_tr:] = bl2.transform(hours[n_tr:], data["dt_arr"][n_tr:], data["rb_arr"][n_tr:],
                          which="eval")
check("条件基线可复现（重算结果一致）", np.allclose(b2, data["B"], atol=1e-3),
      f"max diff = {np.abs(b2 - data['B']).max():.4f}")

# 强泄漏检验：把测试段需求全部置零，测试段基线必须完全不变
y_zero = data["pooled"].copy()
y_zero[n_tr:] = 0.0
bl3 = ConditionalBaseline(pg, config.BASELINE_K, config.BASELINE_LEVEL_WINDOW)
bl3.fit(data["pooled"][core], hours[core], data["dt_arr"][core], data["rb_arr"][core])
bl3.compute_rolling(y_zero, hours, data["dt_arr"], data["rb_arr"],
                    freeze_idx=n_tr, deploy_idx=n)
check("测试段观测不参与基线校准（置零测试段需求后测试段基线不变）",
      np.array_equal(bl3.snapshot_eval[0], bl2.snapshot_eval[0])
      and np.array_equal(bl3.snapshot_eval[4], bl2.snapshot_eval[4]))

# ------------------------------------------------------------------ 3
section("3. 训练—推理分布一致性（气象序列）")
profiles = load_scenario_profiles()
check("情景廓线文件存在", profiles is not None,
      f"情景: {list(profiles.keys()) if profiles else None}")
h24 = pd.date_range("2021-07-05 00:00", periods=24, freq="h")
var_ok, detail = True, []
for sc in ["晴天", "阴天", "小雨", "大雨", "高温"]:
    w = scenario_hourly_weather(sc, h24, profiles)
    v = float(np.var(w[:, 0]))
    detail.append(f"{sc}:{v:.2f}")
    if v <= 1e-9:
        var_ok = False
check("各情景温度通道逐小时变化（非全天常数）", var_ok, "温度方差 " + " ".join(detail))
pr = {sc: float(scenario_hourly_weather(sc, h24, profiles)[0, 1])
      for sc in ["晴天", "阴天", "小雨", "大雨"]}
check("情景语义正确（晴天/阴天零降水，大雨最大）",
      pr["晴天"] == 0.0 and pr["阴天"] == 0.0 and pr["大雨"] > pr["小雨"] > 0,
      str({k: round(v, 2) for k, v in pr.items()}))
check("不同情景产生不同特征",
      not np.allclose(make_features(h24, scenario="晴天"),
                      make_features(h24, scenario="大雨")))

# ------------------------------------------------------------------ 4~9
section("4~9. 已训练模型核查")
models = pt.load_models()
if models is None:
    check("模型文件存在", False, f"{pt.MODEL_PATH} 不存在，请先运行 src/period_trainer.py --force")
else:
    meta = models["meta"]
    check("模型文件存在", True, meta["trained_at"])
    check("主线四模型全部载入",
          {"gbt", "transformer", "lstm", "convlstm"}.issubset(models["results"]),
          str(sorted(models["results"].keys())))
    check("MLP 消融权重仍可载入（代码与权重未物理删除）",
          "mlp" in models["results"], str(sorted(models["results"].keys())))
    check("架构标注为「条件基线 + 残差学习」", meta.get("architecture") == "条件基线 + 残差学习")

    section("4. 同口径对照")
    sizes = {k: m["n_hours"] for k, m in meta["metrics"].items() if k.endswith("|cold")}
    check("所有模型测试小时数完全一致且等于测试段长度",
          len(set(sizes.values())) == 1 and set(sizes.values()) == {len(test)},
          str(sizes))
    check("测试集为时间尾段（与训练/验证不交叉）",
          test[0] == n_tr and test[-1] == n - 1,
          f"[{test[0]}, {test[-1]}] n_tr={n_tr}")

    section("4b. 切分口径与统计量自洽")
    n_core, n_val = data["n_core"], data["n_val"]
    check("切分严格连续无重叠",
          n_core + n_val == n_tr and n_tr + len(test) == n,
          f"core {n_core} + val {n_val} = {n_tr}；holdout {n_tr} + test {len(test)} = {n}")
    check("测试段占比 = 1 - TRAIN_HOLDOUT_RATIO",
          abs(len(test) / n - (1 - config.TRAIN_HOLDOUT_RATIO)) < 0.01,
          f"{len(test)/n:.4f} vs {1-config.TRAIN_HOLDOUT_RATIO:.4f}")
    check("验证段占比 = 留出段 × (1 - TRAIN_CORE_RATIO)",
          abs(n_val / n_tr - (1 - config.TRAIN_CORE_RATIO)) < 0.01,
          f"{n_val/n_tr:.4f} vs {1-config.TRAIN_CORE_RATIO:.4f}")
    eff = meta.get("split", {}).get("effective_ratio")
    check("meta 记录等效占比（供论文与界面引用）",
          isinstance(eff, dict) and abs(eff["core"] - n_core / n) < 1e-4
          and abs(eff["test"] - len(test) / n) < 1e-4,
          str(eff))
    check("meta 记录两处硬编码已外置为配置",
          meta.get("suggest_retrain_days") == config.SUGGEST_RETRAIN_DAYS,
          f"suggest_retrain_days={meta.get('suggest_retrain_days')}")
    comp = meta.get("model_complexity", {})
    check("模型规模无哨兵值（-1 已改为 None）",
          all(v is None or v > 0 for v in comp.values()) and -1 not in comp.values(),
          str(comp))
    check("meta 记录 GBT 超参搜索开关与最优参数",
          "gbt_search_enabled" in meta
          and isinstance(meta.get("gbt_best_params"), dict)
          and {"max_depth", "learning_rate", "max_iter"} <= set(meta["gbt_best_params"]),
          f"search={meta.get('gbt_search_enabled')} params={meta.get('gbt_best_params')}")

    section("4c. 模型文件体积与加载校验")
    import os as _os
    size_mb = _os.path.getsize(pt.MODEL_PATH) / 1024 / 1024
    check("pkl 不含中间大数组（体积 < 5 MB）", size_mb < 5.0, f"{size_mb:.2f} MB")
    check("load_models 返回的 stat 通过形状校验（无异常即通过）",
          models["stat"]["static_map"].shape == (pg, pg)
          and models["stat"]["r_mu"].size == pg * pg,
          f"static_map {models['stat']['static_map'].shape}")

    section("4d. 静态一致性（防止死代码与冗余参数回归）")
    import ast as _ast

    def _tree(rel):
        with open(os.path.join(ROOT, rel), encoding="utf-8-sig") as f:
            return _ast.parse(f.read())

    def _imported(tree):
        names = set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ImportFrom):
                names.update(a.asname or a.name for a in node.names)
            elif isinstance(node, _ast.Import):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
        return names

    dead_in_app = {"pool_3d", "unpool"} & _imported(_tree("app/app.py"))
    check("app.py 不再导入未使用的 pool_3d / unpool", not dead_in_app, str(sorted(dead_in_app)))
    check("period_trainer.py 不再导入未使用的 PEAK_HOURS",
          "PEAK_HOURS" not in _imported(_tree("src/period_trainer.py")))
    check("config 已移除无引用的 WINDOW_SIZE", not hasattr(config, "WINDOW_SIZE"))
    _pt = _tree("src/period_trainer.py")
    _bs = next(n for n in _ast.walk(_pt)
               if isinstance(n, _ast.FunctionDef) and n.name == "build_sequences")
    _args = [a.arg for a in _bs.args.args]
    check("build_sequences 签名不含未使用的 B / y",
          "B" not in _args and "y" not in _args, str(_args))

    _old_flag = config.GBT_SEARCH
    config.GBT_SEARCH = False
    _short = pt._gbt_search({}, tag="静态检查")
    config.GBT_SEARCH = _old_flag
    check("GBT_SEARCH=False 时短路返回缺省参数（不触碰数据、不搜索）",
          all(_short[k] == config.XGB_PARAMS[k] for k in _short), str(_short))

    _bl = ConditionalBaseline(pg, config.BASELINE_K, config.BASELINE_LEVEL_WINDOW)
    _bl.rho = np.zeros((24, 4))
    try:
        _bl.to_dict()
        _raised = False
    except RuntimeError:
        _raised = True
    check("基线未生成冻结快照时 to_dict 立即报错（不静默写出 None）", _raised)

    preds_cold = pt.predict_all(data, models["results"], test, mode="cold")
    preds_roll = pt.predict_all(data, models["results"], test, mode="rolling")
    y_true = data["pooled"][test]

    section("5. 冷启动信息对等")
    ks = dict(z=data["z"], R=data["R"].reshape(n, -1),
              r_mu=data["r_mu"], r_sd=data["r_sd"], static_map=data["static_map"],
              seq_len=24, pg=pg)
    seq_c = pt.build_sequences(indices=test[:16], use_history=False, **ks)[0]
    seq_r = pt.build_sequences(indices=test[:16], use_history=True, **ks)[0]
    F = len(FEATURE_NAMES)
    check("冷启动：残差通道全为 0", np.allclose(seq_c[..., F:F + P], 0))
    check("冷启动：掩码通道全为 0", np.allclose(seq_c[..., F + P], 0))
    check("冷启动：外生特征 z 完整保留", not np.allclose(seq_c[..., :F], 0))
    check("冷启动：ConvLSTM 静态空间先验图保留",
          not np.allclose(pt.build_sequences(indices=test[:16], use_history=False, **ks)[2][..., 2, :, :], 0))
    check("滚动：残差通道非零（历史被真正使用）", not np.allclose(seq_r[..., F:F + P], 0))

    section("6. 指标可复现")
    for key in sorted(preds_cold):
        if f"{key}|cold" not in meta["metrics"]:
            continue      # 未参与本轮训练的内部对照（如 MLP），无官方指标可比对
        m_rec = meta["metrics"][f"{key}|cold"]
        m_re = core_metrics(y_true.ravel(), preds_cold[key].ravel())
        check(f"{key} 冷启动指标可复现",
              abs(m_rec["rmse"] - m_re["rmse"]) < 1e-6
              and abs(m_rec["r2"] - m_re["r2"]) < 1e-9,
              f"RMSE {m_rec['rmse']:.2f}（重算 {m_re['rmse']:.2f}）")

    section("7. 模型有效性")
    base_rmse = float(np.sqrt(np.mean((y_true.ravel()
                                       - data["B"][test].ravel()) ** 2)))
    rmses = {}
    for key in sorted(preds_cold):
        rmses[key] = float(np.sqrt(np.mean((y_true.ravel() - preds_cold[key].ravel()) ** 2)))
        leg = LEGACY.get(key)
        if leg is not None:
            check(f"{key} 优于改造前同族模型", rmses[key] < leg["rmse"],
                  f"RMSE {rmses[key]:.1f} < {leg['rmse']:.1f}")
        check(f"{key} 冷启动不显著劣于条件基线（≤3%）", rmses[key] <= base_rmse * 1.03,
              f"RMSE {rmses[key]:.1f} vs 基线 {base_rmse:.1f}")
    check("至少一个模型冷启动严格优于条件基线",
          any(v < base_rmse for v in rmses.values()),
          f"最优 {min(rmses.values()):.1f} < 基线 {base_rmse:.1f}")
    for key in ("lstm", "convlstm"):
        r_c, r_r = rmses[key], float(np.sqrt(np.mean(
            (y_true.ravel() - preds_roll[key].ravel()) ** 2)))
        check(f"{key} 滚动口径优于冷启动（序列机制产生增益）", r_r < r_c * 0.99,
              f"RMSE {r_r:.1f} < {r_c:.1f}（-{100*(1-r_r/r_c):.1f}%）")
    # 多模型一致性：改造前同一时刻三模型总量相差 41%~69%
    tot = {k: float(p.sum()) for k, p in preds_cold.items()}
    spread = (max(tot.values()) - min(tot.values())) / np.mean(list(tot.values()))
    check("多模型全网总量一致性（差异 < 25%，改造前为 41%~69%）", spread < 0.25,
          " " + " ".join(f"{k}={v:,.0f}" for k, v in tot.items()) + f" | 相对差 {spread:.1%}")

    section("8. 机制生效核查（对照原缺陷 B）")
    sp = {k: float(p.std(axis=(1, 2)).mean()) for k, p in preds_cold.items()}
    check("ConvLSTM 输出空间非均匀（空间卷积有效）", sp["convlstm"] > 10,
          f"空间 std 均值 = {sp['convlstm']:.2f}")
    same = np.allclose(preds_cold["convlstm"], preds_cold["lstm"], atol=1e-3)
    check("ConvLSTM 与 LSTM 输出不再逐值相同（原两者逐值一致）", not same,
          f"平均绝对差 = {np.abs(preds_cold['convlstm']-preds_cold['lstm']).mean():.2f}")
    check("LSTM/ConvLSTM 冷启动 ≠ 仅输出基线（存在残差修正能力）",
          not np.allclose(preds_cold["convlstm"], data["B"][test], atol=1e-4))

    section("9. 预测接口")
    gs = config.GRID_SIZE
    p1 = pt.predict_hour("2026-10-15", 8, "晴天", model_key="gbt")
    check("predict_hour 形状与非负", p1 is not None and p1.shape == (gs, gs)
          and bool((p1 >= 0).all()), None if p1 is None else str(p1.shape))
    check("predict_hour 确定性（同输入同输出）",
          np.allclose(p1, pt.predict_hour("2026-10-15", 8, "晴天", model_key="gbt")))
    p_rain = pt.predict_hour("2026-10-15", 8, "大雨", model_key="gbt")
    check("对天气情景敏感", not np.allclose(p1, p_rain),
          f"晴天 {p1.sum():,.0f} vs 大雨 {p_rain.sum():,.0f}")
    p_we = pt.predict_hour("2026-10-17", 8, "晴天", model_key="gbt")
    check("对日期类型敏感（周四 vs 周六）", not np.allclose(p1, p_we),
          f"周四 {p1.sum():,.0f} vs 周六 {p_we.sum():,.0f}")
    p_hol = pt.predict_hour("2026-10-15", 8, "晴天", model_key="gbt", holiday_override=True)
    check("对节假日覆盖敏感", not np.allclose(p1, p_hol),
          f"强制假日 {p_hol.sum():,.0f}")
    ev = {"type": "activity", "x": 10, "y": 10, "radius": 3, "strength": 0.3}
    p_ev = pt.predict_hour("2026-10-15", 8, "晴天", event=ev, model_key="gbt")
    check("对事件扰动敏感（活动提升局部需求）", p_ev.sum() > p1.sum(),
          f"+{100*(p_ev.sum()/p1.sum()-1):.2f}%")
    p_night = pt.predict_hour("2026-10-15", 3, "晴天", model_key="gbt")
    check("昼夜节律正确（08:00 需求 > 03:00）", p1.sum() > p_night.sum(),
          f"08:00 {p1.sum():,.0f} vs 03:00 {p_night.sum():,.0f}")
    pd_all = pt.predict_day("2026-10-15", "晴天", model_key="gbt")
    check("predict_day 形状 (24, gs, gs)",
          pd_all is not None and pd_all.shape == (24, gs, gs))
    far = pt.predict_hour("2080-06-01", 8, "晴天", model_key="gbt")
    check("远期日期可预测（60 年后）", far is not None and far.shape == (gs, gs))
    holiday_preds = {}
    for mk in ["transformer", "lstm", "convlstm"]:
        p = pt.predict_hour("2026-10-15", 8, "晴天", model_key=mk)
        check(f"predict_hour[{mk}] 正常", p is not None and p.shape == (gs, gs)
              and bool((p >= 0).all()), None if p is None else f"总 {p.sum():,.0f}")
    # 2026 年国庆（2025 后无调休数据，走公历推断路径）
    for d in ("2026-10-01", "2026-10-05"):
        p = pt.predict_hour(d, 8, "晴天", model_key="convlstm")
        holiday_preds[d] = float(p.sum())
    check("2026 年国庆可预测（覆盖 2025 后节假日推断路径）",
          all(v > 0 for v in holiday_preds.values()),
          str({k: f"{v:,.0f}" for k, v in holiday_preds.items()}))

# ------------------------------------------------------------------ 10
section("10. Streamlit 应用冒烟测试")
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app", "app.py"), default_timeout=300)
    at.run()
    check("应用无异常", len(at.exception) == 0,
          "; ".join(e.message for e in at.exception[:3]))
    opts = [s for s in at.sidebar.selectbox if "模型" in s.label]
    check("模型下拉含四项（含 Transformer，MLP 已下线）",
          bool(opts) and len(list(opts[0].options)) == 4
          and "Transformer" in list(opts[0].options),
          None if not opts else str(list(opts[0].options)))
except Exception as e:
    check("应用冒烟测试可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

# ------------------------------------------------------------------ 汇总
section("汇总")
passed = sum(1 for _, ok, _ in RESULTS if ok)
print(f"通过 {passed} / {len(RESULTS)}")
if FAILED:
    print("失败项:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("全部通过")
