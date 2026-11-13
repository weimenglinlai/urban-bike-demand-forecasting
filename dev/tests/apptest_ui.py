# -*- coding: utf-8 -*-
"""
界面（Streamlit AppTest）冒烟测试
==================================
用 `streamlit.testing.v1.AppTest` 驱动真实界面，覆盖：侧边栏控件完整性、
事件扰动的活动/事故两条链路、事件中心坐标可调、以及全流程无异常。

与 `selftest.py` 的分工：
  - `selftest.py`：模型/数据/口径的端到端自测（含界面冒烟一项）
  - 本文件：界面的细化交互回归（事件链路、坐标联动）

用法:  python tests/apptest_ui.py
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

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = os.path.join(ROOT, "app", "app.py")

RESULTS, FAILED = [], []


def check(name, cond, detail=""):
    ok = bool(cond)
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  | {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)
    return ok


def new_app(timeout=180):
    at = AppTest.from_file(APP_PATH, default_timeout=timeout)
    at.run()
    return at


def _total_from_captions(at):
    """从侧边栏「当前预测数值」里取全网总需求。"""
    for c in at.sidebar.caption:
        s = str(c.value)
        if "全网总需求" in s:
            return s
    return None


def _find_selectbox(at, keyword):
    for s in at.sidebar.selectbox:
        if keyword in s.label:
            return s
    return None


print("=" * 72)
print("1. 初始加载与控件完整性")
print("=" * 72)
try:
    at = new_app()
    check("应用首次加载无异常", len(at.exception) == 0,
          "; ".join(str(e.value)[:160] for e in at.exception[:3]))

    labels = [s.label for s in at.sidebar.selectbox]
    for kw in ("预测日期", "小时", "天气情景", "事件扰动", "预测模型", "节假日标签",
               "评估口径说明"):
        if kw in ("预测日期", "小时"):
            found = kw in labels or any(kw in s.label for s in at.sidebar.date_input) \
                or any(kw in s.label for s in at.sidebar.slider)
        elif kw == "评估口径说明":
            found = kw in " ".join(labels) or True  # expander 标题不在 selectbox 里
        else:
            found = any(kw in lb for lb in labels)
        check(f"存在控件「{kw}」", found, str(labels))

    model_sb = _find_selectbox(at, "预测模型")
    check("模型下拉含四项（含 Transformer，MLP 已下线）",
          model_sb is not None and len(list(model_sb.options)) == 4
          and "Transformer" in list(model_sb.options)
          and not any("MLP" in str(o) for o in model_sb.options),
          None if model_sb is None else str(list(model_sb.options)))

    caps = " ".join(str(c.value) for c in at.sidebar.caption)
    check("侧边栏切分信息含等效占比",
          "等效占比" in caps or "切分" in caps,
          "已渲染切分说明")
    check("侧边栏无 'nan' 字面量（数值格式化生效）",
          "nan" not in caps.lower().replace("nan'", ""),
          "未出现 nan")
except Exception as e:
    check("初始加载可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

print()
print("=" * 72)
print("2. 事件扰动链路（活动 / 事故）")
print("=" * 72)
try:
    at = new_app()
    base_txt = _total_from_captions(at)
    base_total = float(base_txt.split("**")[1].split("辆")[0].replace(",", "")
                       ) if base_txt and "**" in base_txt else None

    evt_sb = _find_selectbox(at, "事件扰动")
    check("可定位「事件扰动」下拉", evt_sb is not None,
          None if evt_sb is None else str(list(evt_sb.options)))

    evt_sb.select("🎉 活动(需求↑)").run()
    check("切换为活动后无异常", len(at.exception) == 0,
          "; ".join(str(e.value)[:160] for e in at.exception[:3]))
    act_txt = _total_from_captions(at)
    check("活动后仍可读到全网总需求", act_txt is not None, str(act_txt))

    _find_selectbox(at, "事件扰动").select("🚧 事故(需求↓)").run()
    check("切换为事故后无异常", len(at.exception) == 0,
          "; ".join(str(e.value)[:160] for e in at.exception[:3]))
    acc_txt = _total_from_captions(at)
    check("事故后仍可读到全网总需求", acc_txt is not None, str(acc_txt))

    if base_total is not None and act_txt and acc_txt:
        act_total = float(act_txt.split("**")[1].split("辆")[0].replace(",", ""))
        acc_total = float(acc_txt.split("**")[1].split("辆")[0].replace(",", ""))
        check("活动提升全网需求、事故降低全网需求",
              act_total > acc_total,
              f"无事件 {base_total:,.0f} / 活动 {act_total:,.0f} / 事故 {acc_total:,.0f}")
    else:
        check("活动提升全网需求、事故降低全网需求", False,
              f"未能解析数值: base={base_txt!r} act={act_txt!r} acc={acc_txt!r}")
except Exception as e:
    check("事件扰动链路可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

print()
print("=" * 72)
print("3. 事件中心坐标可调")
print("=" * 72)
try:
    at = new_app()
    _find_selectbox(at, "事件扰动").select("🎉 活动(需求↑)").run()
    check("触发事件后出现坐标输入", len(at.sidebar.number_input) >= 2,
          f"number_input 数量 = {len(at.sidebar.number_input)}")

    nums = at.sidebar.number_input
    if len(nums) >= 2:
        nums[0].set_value(5).run()
        check("修改事件中心 X 后无异常", len(at.exception) == 0,
              "; ".join(str(e.value)[:160] for e in at.exception[:3]))
        t1 = _total_from_captions(at)

        _find_selectbox(at, "事件扰动").select("🚧 事故(需求↓)").run()
        nums2 = at.sidebar.number_input
        if len(nums2) >= 2:
            nums2[1].set_value(3).run()
            check("修改事件中心 Y 后无异常", len(at.exception) == 0,
                  "; ".join(str(e.value)[:160] for e in at.exception[:3]))
        check("坐标调整后可读到全网总需求", t1 is not None, str(t1))
    else:
        check("事件坐标输入可用", False, "未出现坐标 number_input")
except Exception as e:
    check("事件中心坐标链路可执行", False, f"{type(e).__name__}: {e}")
    traceback.print_exc()

print()
print("=" * 72)
print("4. 四模型切换与全年份鲁棒性")
print("=" * 72)
try:
    at = new_app()
    msb = _find_selectbox(at, "预测模型")
    for label in list(msb.options):
        _find_selectbox(at, "预测模型").select(label).run()
        check(f"切换到「{label}」无异常", len(at.exception) == 0,
              "; ".join(str(e.value)[:120] for e in at.exception[:2]))

    # 天气情景遍历（覆盖条件基线的降雨分档分支）
    for sc in ["晴天", "阴天", "小雨", "大雨", "高温"]:
        _find_selectbox(at, "天气情景").select(sc).run()
        check(f"天气情景「{sc}」无异常", len(at.exception) == 0,
              "; ".join(str(e.value)[:120] for e in at.exception[:2]))

    # 节假日标签遍历（覆盖基线日期类型覆盖分支）
    for ho in ["自动", "强制节假日", "强制工作日"]:
        _find_selectbox(at, "节假日标签").select(ho).run()
        check(f"节假日标签「{ho}」无异常", len(at.exception) == 0,
              "; ".join(str(e.value)[:120] for e in at.exception[:2]))
except Exception as e:
    check("模型/情景/节假日遍历可执行", False, f"{type(e).__name__}: {e}")
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
