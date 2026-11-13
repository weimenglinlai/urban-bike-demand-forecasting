# -*- coding: utf-8 -*-
"""
滚动口径（Rolling Mode）界面回归测试
====================================
覆盖 dual-mode 推理链路的五件事：
  1. 冷启动为默认口径，日期不受限
  2. 切到滚动口径后日期被限制在留出测试期，天气/节假日标签被锁定（取真实值）
  3. 滚动口径确实读到了 seq_len 小时历史，并给出可见提示
  4. 界面数值与后端 get_rolling_context + predict_hour 的结果**逐格一致**
  5. 两种口径的缓存互不覆盖（cold → rolling → cold 能回到原值）

用法:  python _dev/tests/apptest_rolling.py
"""

import io
import os
import re
import sys
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _find_project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if os.path.exists(os.path.join(cand, "config.py")) and \
           os.path.isdir(os.path.join(cand, "src")):
            return cand
    return os.path.dirname(os.path.dirname(here))


ROOT = _find_project_root()
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = os.path.join(ROOT, "app", "app.py")
MODE_COLD = "冷启动（无历史推演）"
MODE_ROLL = "滚动口径（高精度短时）"

RESULTS, FAILED = [], []


def check(name, cond, detail=""):
    ok = bool(cond)
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  | {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)
    return ok


def new_app(timeout=600):
    at = AppTest.from_file(APP_PATH, default_timeout=timeout)
    at.run()
    return at


def set_mode(at, mode):
    at.sidebar.radio[0].set_value(mode).run()
    return at


def set_model(at, label):
    for s in at.sidebar.selectbox:
        if "预测模型" in s.label:
            s.set_value(label).run()
            return at
    raise AssertionError("未找到「预测模型」下拉")


def total_of(at):
    """从侧边栏「当前预测数值」里取全网总需求（整数）。"""
    for c in at.sidebar.caption:
        m = re.search(r"全网总需求\(\d\d:00\): \*\*([\d,]+)\s*辆\*\*", str(c.value))
        if m:
            return int(m.group(1).replace(",", ""))
    raise AssertionError("未读到全网总需求")


def err(at):
    return "; ".join(str(e.value)[:200] for e in at.exception[:2])


print("=" * 72)
print("1. 默认口径与控件状态")
print("=" * 72)
try:
    at = new_app()
    check("首次加载无异常", len(at.exception) == 0, err(at))
    check("存在「推理口径」单选按钮",
          len(at.sidebar.radio) >= 1 and "推理口径" in at.sidebar.radio[0].label)
    check("默认为冷启动口径", at.sidebar.radio[0].value == MODE_COLD, at.sidebar.radio[0].value)
    di = at.sidebar.date_input[0]
    # Streamlit 的 date_input 默认给 ±10 年（2016~2036），不含 2021 即说明未按历史区间收紧
    check("冷启动下日期未被限制在历史区间",
          di.min is None or di.min.year < 2021,
          f"min={di.min} max={di.max}")
    tot_cold_gbt = total_of(at)
    check("冷启动可出预测", tot_cold_gbt > 0, f"{tot_cold_gbt} 辆")
except Exception as e:
    check("默认口径检查可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


print()
print("=" * 72)
print("2. 切换到滚动口径：日期收窄 + 情景锁定")
print("=" * 72)
try:
    set_mode(at, MODE_ROLL)
    check("切换滚动口径无异常", len(at.exception) == 0, err(at))

    di = at.sidebar.date_input[0]
    ok_win = (di.min is not None and di.max is not None
              and str(di.min) >= "2021-07-14" and str(di.max) <= "2021-07-31")
    check("日期被限制在留出测试期内", ok_win, f"{di.min} ~ {di.max}（value={di.value}）")

    wsb = [s for s in at.sidebar.selectbox if "天气情景" in s.label][0]
    hsb = [s for s in at.sidebar.selectbox if "节假日标签" in s.label][0]
    check("滚动口径下天气情景被锁定（取真实值）", wsb.disabled is True)
    check("滚动口径下节假日标签被锁定（取真实值）", hsb.disabled is True)

    # 只有序列模型才会真正读取历史，先切到 ConvLSTM 再看提示
    set_model(at, "ConvLSTM")
    succ = [s.value for s in at.sidebar.success if "已自动读取" in str(s.value)]
    check("侧边栏出现「已自动读取 …」提示", len(succ) == 1, succ[0] if succ else "无")
    check("提示中的小时数与模型 seq_len 一致（24h）",
          bool(succ) and "24 小时" in succ[0], succ[0] if succ else "")
except Exception as e:
    check("滚动口径控件检查可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


print()
print("=" * 72)
print("3. 界面数值 = 后端滚动推理（逐格一致）")
print("=" * 72)
try:
    import numpy as np
    import src.period_trainer as pt

    models = pt.load_models()
    set_model(at, "ConvLSTM")
    at.sidebar.date_input[0].set_value(__import__("datetime").date(2021, 7, 31)).run()
    at.sidebar.slider[0].set_value(8).run()
    check("滚动 + ConvLSTM 无异常", len(at.exception) == 0, err(at))

    ui_total = total_of(at)
    ctx = pt.get_rolling_context("2021-07-31 08:00", "convlstm", models)
    p = pt.predict_hour("2021-07-31", 8, model_key="convlstm", models=models,
                        recent_residual=ctx["residual"],
                        rolling_weather={"norm": ctx["weather_norm"],
                                         "precip_mm": ctx["precip_mm"],
                                         "hours": ctx["hours"]},
                        baseline_which="eval")
    back_total = int(np.ceil(p.sum()))
    check("界面总数与后端一致", ui_total == back_total,
          f"UI={ui_total} 后端={back_total}")
    check("滚动结果与冷启动结果不同（确认历史真的被用上）",
          ui_total != tot_cold_gbt, f"滚动={ui_total} 冷启动(GBT)={tot_cold_gbt}")

    # 真实值对比：滚动误差应小于冷启动误差
    t, hours, _gm = pt.build_hourly_demand()
    i = int(np.where(__import__("pandas").DatetimeIndex(hours)
                     == __import__("pandas").Timestamp("2021-07-31 08:00"))[0][0])
    truth = float(np.asarray(t)[i].sum())
    p_cold = pt.predict_hour("2021-07-31", 8, model_key="convlstm", models=models)
    e_roll = abs(p.sum() - truth)
    e_cold = abs(p_cold.sum() - truth)
    check("该时刻滚动误差小于冷启动误差", e_roll < e_cold,
          f"真实={truth:.0f} 滚动误差={e_roll:.0f} 冷启动误差={e_cold:.0f}")
except Exception as e:
    check("界面/后端一致性校验可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


print()
print("=" * 72)
print("4. 无序列结构的模型：滚动 == 冷启动")
print("=" * 72)
try:
    set_model(at, "梯度提升树")
    check("GBT + 滚动口径无异常", len(at.exception) == 0, err(at))
    gbt_roll = total_of(at)
    info_txt = " ".join(str(i.value) for i in at.sidebar.info)
    check("给出「无序列结构、与冷启动一致」提示",
          "无序列结构" in info_txt, info_txt[:80])

    set_mode(at, MODE_COLD)
    at.sidebar.date_input[0].set_value(__import__("datetime").date(2021, 7, 31)).run()
    at.sidebar.slider[0].set_value(8).run()
    gbt_cold = total_of(at)
    check("GBT 滚动与冷启动数值完全相同", gbt_roll == gbt_cold,
          f"滚动={gbt_roll} 冷启动={gbt_cold}")
except Exception as e:
    check("GBT 口径等价性校验可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


print()
print("=" * 72)
print("5. 缓存隔离：cold → rolling → cold 不串味")
print("=" * 72)
try:
    # 回到冷启动、ConvLSTM、今天 08:00
    set_model(at, "ConvLSTM")
    today = __import__("datetime").date.today()
    at.sidebar.date_input[0].set_value(today).run()
    at.sidebar.slider[0].set_value(8).run()
    first_cold = total_of(at)

    set_mode(at, MODE_ROLL)          # 切滚动（同一模型，日期被拉到测试期）
    at.sidebar.date_input[0].set_value(__import__("datetime").date(2021, 7, 20)).run()
    at.sidebar.slider[0].set_value(18).run()
    roll_val = total_of(at)

    set_mode(at, MODE_COLD)          # 切回冷启动
    at.sidebar.date_input[0].set_value(today).run()
    at.sidebar.slider[0].set_value(8).run()
    second_cold = total_of(at)

    check("冷启动结果在切换口径后保持不变（缓存未串味）",
          first_cold == second_cold and roll_val != first_cold,
          f"cold1={first_cold} rolling={roll_val} cold2={second_cold}")
except Exception as e:
    check("缓存隔离校验可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()


print()
print("=" * 72)
print("汇总")
print("=" * 72)
passed = sum(1 for _, ok, _ in RESULTS if ok)
print(f"通过 {passed} / {len(RESULTS)}")
if FAILED:
    print("失败项:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print("全部通过")
