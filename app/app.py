"""
共享单车需求预测与调度系统 - Streamlit 可视化应用
====================================================
项目核心交互界面。

两种推理口径（侧边栏顶部切换）:
  * 冷启动（默认）--只输入外生特征(小时/星期几/节假日/调休/天气),可预测任意未来日期,
    无递归、无需历史需求观测;四个模型输入信息完全相同,构成严格同口径对照。
  * 滚动口径--额外读取目标时刻前 seq_len 小时的真实残差序列与真实气象,
    仅对历史中已有观测的时刻可用(限制在留出测试期内),
    用于展示序列/时空机制在信息可得时的增益。

设计原则:
1. 冷启动口径纯未来预测;滚动口径是历史回放演示,二者在界面上明确区分、互不混淆
2. 四个模型用「外生特征 + 条件基线」训练, 目标为残差 r(t) = y(t) − B(t)
3. 冷启动日期默认今天、无上限;滚动口径限制在测试期以保证无未来信息泄漏
4. 精准节假日:chinese_calendar(2004-2025)+ 公历推断(2025 后)+ 周末 + 调休上班日
5. 调度建议基于预测需求(不依赖库存):高需求→投放建议,低需求→回收建议
6. 渐变色块:网格颜色从绿(低需求)渐变到红(高需求),支持点击选中并联动曲线
"""

import sys, os, subprocess, re, io, base64, pickle, time
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="共享单车需求预测与调度",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══ 全局布局微调(亮色主题下适配) ═══
st.markdown(
    """<style>
    /* 主标题:压缩字号避免换行成两行,消除大段空白 */
    section:not([data-testid="stSidebar"]) h1 {
        font-size: 26px !important;
        line-height: 1.4 !important;
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
    }
    /* 压缩副标题与子标题间距 */
    .main h3 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    /* 图例颜色在亮色背景更清晰 */
    .main .caption {
        color: #555 !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

from PIL import Image as _PIL_Img
from PIL import ImageFilter as _PIL_ImageFilter
import config
from src.date_features import is_holiday, holiday_name, make_features
from src.period_trainer import (
    predict_hour,
    predict_day,
    get_model_info,
    get_rolling_context,
    get_rolling_range,
    MODEL_PATH as PT_MODEL_PATH,
)
gs = config.GRID_SIZE
POOL = config.XGB_POOL

# 具备滚动口径的模型：只有序列 / 注意力模型有历史残差通道，GBT 没有
ROLLING_MODELS = ("lstm", "convlstm", "transformer")
# 内部保留但不在主线展示的模型：MLP 已完成「隔离序列机制贡献」的消融使命，
# 由 Transformer 接替其位置；其代码、权重与历史结论全部保留（见 README 附录）。
HIDDEN_MODELS = ("mlp",)
pg = gs // POOL  # 5

WEATHER_SCENARIOS = ["晴天", "阴天", "小雨", "大雨", "高温"]

MODEL_LABELS = ["梯度提升树", "LSTM", "ConvLSTM", "Transformer"]


def _fmt_num(v, nd=1):
    """数值格式化：None / NaN / Inf 统一显示为 —，避免界面出现 'nan'。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(f):
        return "—"
    return f"{f:.{nd}f}"


MODEL_KEYS = {"梯度提升树": "gbt", "LSTM": "lstm", "ConvLSTM": "convlstm",
              "Transformer": "transformer"}

# 机制 / 适用 / 不适用：结构性事实（与数据无关），故写成常量；
# 所有会随训练数据变化的数字（RMSE / MAE / R² / 延迟 / 排名 / 推荐度）一律现算。
MODEL_TEXT = {
    "gbt": {
        "mech": "传统机器学习基线：逐区块 HistGradientBoosting，由外生特征直接学残差。",
        "fit": "无 PyTorch 依赖的纯 sklearn 部署；需要可解释性的场景；"
               "作为「机器学习能否超过统计基线」的对照。",
        "unfit": "实时交互（逐区块拟合，推理最慢）；以 MAE 为考核指标的场景。"
                 "无序列结构，滚动口径与冷启动结果完全一致。",
    },
    "lstm": {
        "mech": "时序深度学习：双通道，循环前 seq_len 小时残差序列 + 外生直连分支。",
        "fit": "有近 seq_len 小时真实观测的滚动预测；延迟敏感场景。",
        "unfit": "冷启动口径（无历史可循环，序列机制无从发挥）；"
                 "需空间邻域平滑的场景。界面请切到「滚动口径」再选本模型。",
    },
    "convlstm": {
        "mech": "时空深度学习：在 pg×pg 残差图上卷积，外生条件经 FiLM 逐通道调制卷积门控。",
        "fit": "系统默认首选：纯未来预测、有历史观测时、早晚高峰与全网总量、"
               "通勤走廊等存在空间连片结构的场景。",
        "unfit": "极致的点位级峰值定位（空间卷积会平滑单区块峰值）；延迟极敏感场景。",
    },
    "transformer": {
        "mech": "自注意力深度学习：双通道，序列分支用 Transformer Encoder 对前 "
                "seq_len 小时残差做全局自注意力（带正弦位置编码），"
                "外生直连分支与 LSTM 同构。序列内任意两步直接可见，"
                "长程依赖无需逐小时传递。",
        "fit": "需要建模相隔较远时段的相互影响（如凌晨调度对晚高峰的滞后效应）；"
               "有近 seq_len 小时真实观测的滚动预测；推理延迟要求不苛刻的场景。",
        "unfit": "冷启动口径（无历史残差可注意力，序列分支退化）；"
                 "需空间邻域平滑的场景（自注意力不含空间先验）。"
                 "界面请切到「滚动口径」再选本模型。",
    },
}

# 推荐度权重：最佳口径 R² 0.40 / 冷启动 MAE 0.25 / 冷启动 R² 0.20 / 单日推理延迟 0.15
# 「最佳口径 R²」= 冷启动与滚动中较高者：序列模型在滚动下更强是真实能力，
# 冷启动 R² 单列 20% 权重，保证「无历史也能用」这一点同样被计分。
STAR_W = {"r2_best": 0.40, "mae_cold": 0.25, "r2_cold": 0.20, "latency": 0.15}


def _norm(d, bigger_better=True):
    """min-max 归一到 0~1；全部相同则给 0.5（不参与区分）。"""
    vals = [v for v in d.values() if v is not None]
    if len(vals) < 2:
        return {k: 0.5 for k in d}
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.5 for k in d}
    return {k: ((v - lo) / (hi - lo) if bigger_better else (hi - v) / (hi - lo))
            if v is not None else 0.5 for k, v in d.items()}


def _model_scores(metrics, bench):
    """按当前训练指标计算综合得分与星级（1~5）。

    只依赖 meta 与实测延迟，换数据集重训后自动刷新；指标缺失的模型不参与打分。
    注意：星级只反映指标，不反映稳定性/复现性等工程因素。
    """
    rows = {}
    for k in ("gbt", "transformer", "lstm", "convlstm"):
        mc = metrics.get(f"{k}|cold")
        if not mc:
            continue
        mr = metrics.get(f"{k}|rolling") or mc
        rows[k] = {"r2c": float(mc["r2"]), "maec": float(mc["mae"]),
                   "r2b": max(float(mc["r2"]), float(mr["r2"])),
                   "lat": (bench or {}).get(k)}
    if not rows:
        return {}, {}
    n_b = _norm({k: r["r2b"] for k, r in rows.items()})
    n2 = _norm({k: r["maec"] for k, r in rows.items()}, False)
    n_c = _norm({k: r["r2c"] for k, r in rows.items()})
    has_lat = all(r["lat"] for r in rows.values())
    n4 = _norm({k: r["lat"] for k, r in rows.items()}, False) if has_lat else {}
    score = {}
    for k in rows:
        s = (STAR_W["r2_best"] * n_b[k] + STAR_W["mae_cold"] * n2[k]
             + STAR_W["r2_cold"] * n_c[k]
             + (STAR_W["latency"] * n4[k] if has_lat else 0.0))
        if not has_lat:                      # 无延迟数据时其余权重按比例放大
            s = s / (1 - STAR_W["latency"])
        score[k] = s
    stars = {k: max(1, min(5, 1 + int(round(4 * v)))) for k, v in score.items()}
    return stars, score


def _rank_txt(metrics, k, field="r2"):
    """某模型在冷启动口径下某指标的排名，返回「第 n/m」文本。"""
    vals = {kk: float(v[field]) for kk, v in
            ((x.split("|")[0], y) for x, y in metrics.items() if x.endswith("|cold"))}
    if k not in vals:
        return ""
    order = sorted(vals, key=vals.get, reverse=True)
    return f"第 {order.index(k) + 1}/{len(vals)}"


def _multi_replace(s, mapping):
    """按映射批量替换模板占位词。"""
    for a, b in mapping.items():
        s = s.replace(a, b)
    return s


def _resolve_seq_len_ui():
    """界面展示用的序列长度（与训练侧 _resolve_seq_len 同源）。"""
    try:
        return int(config.LSTM_PARAMS.get("seq_len")
                   or config.CONVLSTM_PARAMS.get("seq_len") or 24)
    except Exception:
        return 24


def _build_model_notes(model_choice, pt_info, bench):
    """按当前训练结果生成模型说明（数字全部来自 meta 与实测延迟）。"""
    key = MODEL_KEYS[model_choice]
    t = MODEL_TEXT.get(key)
    if not t:
        return "当前模型暂无说明文本。"
    # 模板中的 pg / seq_len 是随数据与配置变化的量，渲染时替换为实际数值，
    # 避免把代码变量名直接展示给用户
    _seq_len = _resolve_seq_len_ui()
    _subst = {"pg×pg": f"{pg}×{pg}", "seq_len": f"{_seq_len}"}
    t = {k: _multi_replace(v, _subst) for k, v in t.items()}
    parts = [f"**机制**　{t['mech']}"]

    metrics = ((pt_info or {}).get("metrics") or {})
    mc = metrics.get(f"{key}|cold")
    mr = metrics.get(f"{key}|rolling")
    if mc:
        stars, _ = _model_scores(metrics, bench)
        line = (f"**实测（当前训练结果）**　冷启动 RMSE {_fmt_num(mc['rmse'], 2)}｜"
                f"MAE {_fmt_num(mc['mae'], 2)}｜R² {_fmt_num(mc['r2'], 4)}"
                f"（R² {_rank_txt(metrics, key, 'r2')}）")
        if mr and key in ("lstm", "convlstm"):
            gain = (float(mr["rmse"]) / float(mc["rmse"]) - 1) * 100
            line += (f"；滚动 RMSE {_fmt_num(mr['rmse'], 2)}｜R² {_fmt_num(mr['r2'], 4)}"
                     f"（较冷启动 {gain:+.1f}%）")
        lat = (bench or {}).get(key)
        if lat:
            lat_rank = ""
            if all((bench or {}).get(k) for k in ("gbt", "transformer", "lstm", "convlstm")):
                order = sorted(bench, key=bench.get)
                lat_rank = f"（{order.index(key) + 1}/{len(order)} 快）"
            line += f"；单日 24h 推理 {lat:.0f} ms{lat_rank}"
        parts.append(line)

        seg = mc.get("segments") or {}
        seg_bits = []
        for name in ("高峰时段", "有雨", "无雨", "周末"):
            sv = seg.get(name)
            if sv and sv.get("r2") is not None:
                seg_bits.append(f"{name} R² {_fmt_num(sv['r2'], 4)}")
        if seg_bits:
            parts.append("**细分场景**　" + "｜".join(seg_bits))

    parts.append(f"**适用**　{t['fit']}")
    parts.append(f"**不适用**　{t['unfit']}")
    if mc:
        stars, _ = _model_scores(metrics, bench)
        s = stars.get(key, 0)
        parts.append(f"**推荐度　{'★' * s}{'☆' * (5 - s)}**　"
                     f"按当前指标自动计算（最佳口径 R² {STAR_W['r2_best']:.0%}、"
                     f"MAE {STAR_W['mae_cold']:.0%}、冷启动 R² {STAR_W['r2_cold']:.0%}、"
                     f"延迟 {STAR_W['latency']:.0%}）；换数据重训后自动刷新。"
                     f"星级只反映指标，不含稳定性/复现性等工程因素。")
    return "\n\n".join(parts)

# ═══ 事件中心默认值：全期需求热点（而非几何中心）═══
@st.cache_data(show_spinner=False)
def _demand_hotspot():
    """全期累计需求最高的网格坐标 (x, y)，作为事件中心默认值。

    取真实热点而非几何中心 (10,10)：活动/事故不会恰好发生在网格正中，
    默认落在实际需求最集中的区域更接近现实情形。
    """
    try:
        from src.period_trainer import build_hourly_demand
        t, _h, _gm = build_hourly_demand()
        cum = np.asarray(t).sum(axis=0)
        idx = np.unravel_index(int(np.argmax(cum)), cum.shape)
        return (int(idx[0]), int(idx[1]))
    except Exception:
        return (gs // 2, gs // 2)


# 事件中心真源用普通 session_state（非 widget key）：Streamlit 禁止在 widget 实例化后
# 改写其绑定 key，而地图点击发生在侧边栏输入框之后，故用 ev_center + ev_ver 版本号，
# 点击后递增 ev_ver 使输入框以新 key 重建并读取新值（与 clicked_grid 同一模式）。
if "ev_center" not in st.session_state:
    st.session_state.ev_center = _demand_hotspot()
if "ev_ver" not in st.session_state:
    st.session_state.ev_ver = 0

# ═══ 模型对象缓存（侧边栏的日期范围依赖它，故提前定义）═══
@st.cache_resource(show_spinner=False)
def _cached_models(model_mtime):
    """按模型文件 mtime 缓存的模型对象（跨会话共享）。

    用 cache_resource 而不是 cache_data：模型是 torch Module + sklearn 森林，
    既不便宜也不可安全深拷贝，而推理路径全程只读（eval 模式 + no_grad），
    共享同一实例是安全的。缓存键为 model_mtime，重训后自动重载。
    """
    from src.period_trainer import load_models
    return load_models()


def _model_mtime():
    """模型文件最新修改时间,作为缓存失效信号"""
    import glob
    paths = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "period_models*"))
    if not paths:
        return 0.0
    return max(os.path.getmtime(p) for p in paths)


@st.cache_data(show_spinner=False)
def _bench_latency(model_mtime):
    """实测各模型单日 24h 冷启动推理耗时（ms），按模型文件版本缓存。

    延迟随数据规模、网格大小与硬件变化，写死在代码里必然失真，故实测。
    一个模型版本只测一次（约 3 s），重训后自动重测。
    """
    models = _cached_models(model_mtime)
    if not models:
        return {}
    day = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    out = {}
    for k in ("gbt", "transformer", "lstm", "convlstm"):
        if k not in (models.get("results") or {}):
            continue
        try:
            t0 = time.perf_counter()
            predict_day(day, "晴天", event=None, model_key=k,
                        holiday_override=None, models=models, rolling=False)
            out[k] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception:
            continue
    return out


def _gb(n):
    """字节数转可读文本。"""
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# ═══ 侧边栏 ═══
st.sidebar.title("🚲 控制面板")

# ─── 推理口径：冷启动 / 滚动 ───
MODE_COLD, MODE_ROLL = "冷启动（无历史推演）", "滚动口径（高精度短时）"
mode_label = st.sidebar.radio(
    "推理口径", [MODE_COLD, MODE_ROLL], index=0,
    help="冷启动：只用日历、节假日与气象等外生特征，可预测任意未来日期；"
         f"滚动口径：额外读取目标时刻前 {_resolve_seq_len_ui()} 小时的真实残差，"
         "精度更高，但只对历史数据中已有观测的时段有效。",
)
rolling = (mode_label == MODE_ROLL)
mode = "rolling" if rolling else "cold"

_models_obj = _cached_models(_model_mtime())
_roll_start, _roll_end = get_rolling_range(_models_obj) if rolling else (None, None)

today = pd.Timestamp.now().normalize().date()
_dmin = _dmax = None
if rolling and _roll_start is not None:
    # 只开放整日落在留出测试期内的日期：整日 24 小时都能凑满前 seq_len 小时观测
    _dmin = (_roll_start.normalize() if _roll_start == _roll_start.normalize()
             else _roll_start.normalize() + pd.Timedelta(days=1)).date()
    _dmax = _roll_end.normalize().date()
    date = st.sidebar.date_input(
        "预测日期（滚动口径限历史区间）",
        value=min(max(today, _dmin), _dmax),
        min_value=_dmin, max_value=_dmax,
        help=f"滚动口径需要目标时刻之前的真实观测，故仅开放留出测试期 "
             f"{_dmin} ~ {_dmax}：该段未参与训练、基线冻结在划分点之前，"
             f"与论文滚动指标完全同口径。",
    )
else:
    date = st.sidebar.date_input(
        "预测日期",
        value=today,
        help="选择要预测的日期,支持今天及任意未来日期(含 60 年后)",
    )
hour = st.sidebar.slider("选择小时", 0, 23, 8, format="%02d:00")

weather_mode = st.sidebar.selectbox(
    "天气情景", WEATHER_SCENARIOS, disabled=rolling,
    help="预测时的天气假设;不同天气会显著改变预测结果。"
         + ("滚动口径下气象取历史真实值,此选项不生效。" if rolling else ""),
)
if rolling:
    st.sidebar.caption("🌦️ 滚动口径下天气与节假日标签均取**历史真实值**,"
                       "上面的情景假设不生效。")

event_type = st.sidebar.selectbox(
    "事件扰动", ["无", "🎉 活动(需求↑)", "🚧 事故(需求↓)"], index=0,
    help="叠加区域事件影响:活动使区域内需求上升,事故使需求下降",
)
event = None
if event_type != "无":
    event_strength = st.sidebar.slider("事件强度", 0.1, 1.0, 0.3, 0.1,
                                       help="强度越大,局部需求变化越显著")
    strength = float(event_strength)
    radius = 3 if "活动" in event_type else 2
    # 事件中心:点击地图网格设定,或在此手动微调;二者共享 session_state.ev_center
    _cx, _cy = st.session_state.ev_center
    _ver = st.session_state.ev_ver
    _ex = st.sidebar.number_input(
        "事件中心 X（网格列）", 0, gs - 1, int(_cx), 1, key=f"ev_x_{_ver}",
        help="事件中心所在网格的列坐标；也可直接点击地图上的网格设定",
    )
    _ey = st.sidebar.number_input(
        "事件中心 Y（网格行）", 0, gs - 1, int(_cy), 1, key=f"ev_y_{_ver}",
        help="事件中心所在网格的行坐标；也可直接点击地图上的网格设定",
    )
    if (int(_ex), int(_ey)) != (_cx, _cy):
        st.session_state.ev_center = (int(_ex), int(_ey))
        _cx, _cy = st.session_state.ev_center
    st.sidebar.caption(
        f"📌 事件中心 **({_cx},{_cy})** — 点击地图任一网格即可设为事件中心"
    )
    event = {"type": "activity" if "活动" in event_type else "accident",
             "x": int(_cx), "y": int(_cy), "radius": radius, "strength": strength}

model_choice = st.sidebar.selectbox("预测模型", MODEL_LABELS,
                                    help="四个模型共享同一数据集、同一时间切分、同一特征与同一目标分解。"
                                         "冷启动口径下四者输入信息完全相同；滚动口径仅 "
                                         "LSTM / ConvLSTM / Transformer 有历史残差通道。"
                                         "梯度提升树为传统机器学习基线，LSTM 为时序深度学习，"
                                         "ConvLSTM 为时空深度学习，Transformer 为自注意力深度学习。")
model_key = MODEL_KEYS[model_choice]
if _models_obj is not None and model_key not in (_models_obj.get("results") or {}):
    st.sidebar.warning(f"「{model_choice}」尚无权重（未训练或权重文件缺失）。"
                       f"请点下方「🔄 重新训练模型」后再使用；"
                       f"在此之前请改选其它模型。")
    st.stop()

# 节假日标签手动覆盖: 2025 年后 chinese_calendar 无调休数据,自动推断仅含公历固定节日
hol_override = st.sidebar.selectbox(
    "节假日标签", ["自动", "强制节假日", "强制工作日"], index=0,
    disabled=rolling,
    help="默认自动推断;2025 年后调休安排未知,如需精确模拟可手动指定。"
         + ("滚动口径下日期类型取历史真实值,此选项不生效。" if rolling else ""),
)
holiday_override = {"自动": None, "强制节假日": True, "强制工作日": False}[hol_override]

# ─── 滚动口径：取目标时刻前 seq_len 小时的真实残差 ───
_resid = _wnorm = _precip = _win_hours = None
roll_ctx = None
if rolling:
    _target_ts = pd.Timestamp(f"{date} {hour:02d}:00:00")
    if model_key in ROLLING_MODELS:
        roll_ctx = get_rolling_context(_target_ts, model_key, _models_obj)
        if roll_ctx is None:
            st.sidebar.error(
                f"⚠️ 无法取得 {_target_ts:%Y-%m-%d %H:%M} 之前 "
                f"{config.LSTM_PARAMS['seq_len']} 小时的真实残差——"
                f"目标时刻不在历史数据内，或距序列起点过近。"
                f"请改选滚动口径允许的历史日期，或切换为冷启动口径。"
            )
            st.stop()
        _resid = roll_ctx["residual"]
        _wnorm = roll_ctx["weather_norm"]
        _precip = roll_ctx["precip_mm"]
        _win_hours = roll_ctx["hours"]
        st.sidebar.success(
            f"✅ 已自动读取该时刻前 {roll_ctx['seq_len']} 小时的真实数据："
            f"{roll_ctx['start']:%m-%d %H:%M} ~ {roll_ctx['end']:%m-%d %H:%M}"
        )
    else:
        st.sidebar.info(
            f"{model_choice} 无序列结构（没有历史残差通道），滚动口径与冷启动结果"
            f"完全一致；查看滚动增益请改选 LSTM / ConvLSTM。"
        )

st.sidebar.markdown("---")

# ═══ 模型信息（指标来自 meta，换数据重训后自动刷新）═══
pt_info = get_model_info()
if pt_info:
    with st.sidebar.expander("ℹ️ 模型说明（能力 / 场景 / 推荐度）"):
        st.markdown(_build_model_notes(model_choice, pt_info,
                                       _bench_latency(_model_mtime())))
if pt_info:
    with st.sidebar.expander("⏳ 模型信息 / 实测指标"):
        st.caption(f"当前口径：**{mode_label}**")
        st.caption(f"训练时间: {pt_info['trained_at']}")
        st.caption(f"数据范围: {pt_info['data_range'][0]} ~ {pt_info['data_range'][1]}")
        st.caption(f"小时数: {pt_info['n_hours']}")
        sp = pt_info.get("split")
        if sp:
            eff = sp.get("effective_ratio")
            if eff:
                st.caption(f"切分: 核心 {sp['n_core']} / 验证 {sp['n_val']} / 测试 {sp['n_test']} 小时")
                st.caption(f"等效占比: {eff['core']:.1%} / {eff['val']:.1%} / {eff['test']:.1%}")
            else:
                st.caption(f"切分: 训练 {sp.get('n_core')} / 验证 {sp.get('n_val')} / "
                           f"测试 {sp.get('n_test')} 小时")

        _m = pt_info["metrics"]
        # ① 冷启动：四模型 + 条件基线，同口径同输入
        st.markdown("**① 冷启动口径**（无历史观测，四模型输入完全相同）")
        _cold_rows = []
        for key, mv in _m.items():
            name, md = key.split("|") if "|" in key else (key, "")
            if md not in ("cold", "baseline"):
                continue
            if name in HIDDEN_MODELS:       # 内部消融对照，不在主线展示
                continue
            _cold_rows.append({
                "模型": "条件基线" if md == "baseline" else name.upper(),
                "RMSE": round(float(mv["rmse"]), 2),
                "MAE": round(float(mv["mae"]), 2),
                "R²": round(float(mv["r2"]), 4),
            })
        _cold_rows.sort(key=lambda r: r["模型"])
        st.dataframe(pd.DataFrame(_cold_rows), hide_index=True, width="stretch")

        # ② 滚动：只有序列模型有变化，用降幅直接体现增益
        _sl_hint = ((_models_obj or {}).get("results", {}).get(model_key) or {}).get(
            "seq_len") or config.LSTM_PARAMS["seq_len"]
        st.markdown(f"**② 滚动口径**（额外提供前 {_sl_hint}h 真实残差）")
        _SEQ_MODELS = ("lstm", "convlstm", "transformer")
        _roll_rows = []
        for name in _SEQ_MODELS:
            mc = _m.get(f"{name}|cold")
            mr = _m.get(f"{name}|rolling")
            if not (mc and mr):
                continue
            _roll_rows.append({
                "模型": name.upper(),
                "RMSE": round(float(mr["rmse"]), 2),
                "R²": round(float(mr["r2"]), 4),
                "降幅": f"{(mr['rmse'] / mc['rmse'] - 1) * 100:+.1f}%",
            })
        if _roll_rows:
            st.dataframe(pd.DataFrame(_roll_rows), hide_index=True, width="stretch")
            _best = min(_roll_rows, key=lambda r: r["RMSE"])
            # 绝对降幅放在文字里：窄侧边栏放不下「−135.76（−16.2%）」这种长单元格
            _abs_txt = "、".join(
                f"{n.upper()} −{abs(float(_m[f'{n}|rolling']['rmse'])
                                    - float(_m[f'{n}|cold']['rmse'])):.2f}"
                for n in _SEQ_MODELS
                if f"{n}|rolling" in _m and f"{n}|cold" in _m)
            st.caption(f"↑ **{_best['模型']} 滚动 RMSE 降至 {_best['RMSE']}，"
                       f"降幅 {_best['降幅']}，为全项目最优**；"
                       f"序列 / 时空 / 注意力机制的价值只在此口径体现。")
            if _abs_txt:
                st.caption(f"RMSE 绝对降幅：{_abs_txt}")
        _nosm = [f"{n.upper()} {_fmt_num(_m[f'{n}|cold'].get('rmse'), 2)}"
                 for n in ("gbt",) if f"{n}|cold" in _m]
        if _nosm:
            st.caption("GBT 无序列结构，无历史残差通道，滚动口径与冷启动完全一致"
                       f"（RMSE {' / '.join(_nosm)}），故不单列。")
        st.caption("界面推理端会把负值裁剪为 0，实测 R² 略优于上表（约 +0.001），"
                   "属后处理差异，不改变模型对比结论。")
        st.caption(f"建议重训: 每 {pt_info['suggest_retrain_days']} 天或数据政策重大变化时")
else:
    st.sidebar.warning("⚠️ 模型未训练")

with st.sidebar.expander("📐 评估口径说明"):
    st.caption("**冷启动（cold）**：无历史残差观测，纯未来预测，可预测任意日期；"
               "四个模型信息量完全相同，构成严格同口径对照。")
    st.caption(f"**滚动（rolling）**：额外提供目标时刻前 {config.LSTM_PARAMS['seq_len']}h "
               f"真实残差与真实气象，"
               "只对历史中已有观测的时刻有效，用于量化序列机制的增益。")
    st.caption("**滚动口径为何限在测试期**：该段未参与训练，且条件基线冻结在划分点之前，"
               "既无未来信息泄漏，又与论文滚动指标同口径，可逐项复现。")
    st.caption("**条件基线**：按「小时 × 日期类型 × 降雨档」统计的条件均值，"
               "作为零模型参考。")
    st.caption("**MAPE 覆盖率**：MAPE 仅在真实需求 ≥1 的样本上计算并报告占比；"
               "显示为「—」表示该分组无有效样本。")

# ═══ 训练数据（上传 / 本地登记 / 启停）═══
# 数据集清单是「训练用哪些文件」的唯一真源；界面所有数据描述均由它和模型 meta 生成，
# 不再写死「深圳 2021-04~07」。换数据或加数据后只需在此登记 + 重训。
from src import dataset_registry as dreg

_MAX_UP_MB = int(st.get_option("server.maxUploadSize") or 200)
_ds_sum = dreg.summarize()
_model_ds = (pt_info or {}).get("data_sources") or {}
_fp_now = dreg.fingerprint()

with st.sidebar.expander("📁 训练数据（上传 / 登记 / 启停）"):
    st.caption(f"已登记 **{_ds_sum['n_datasets']}** 个数据集，启用 "
               f"**{_ds_sum['n_active']}** 个（{_ds_sum['n_files']} 个文件，"
               f"{_gb(_ds_sum['total_bytes'])}）")
    if _model_ds.get("fingerprint"):
        if _model_ds["fingerprint"] != _fp_now:
            st.warning("⚠️ 启用数据与当前模型的训练数据不一致，"
                       "当前指标仍属旧数据，重训后才会更新。")
        else:
            st.caption("✅ 当前模型与启用数据一致")
    elif pt_info:
        st.caption("当前模型训练于数据登记功能上线前，未记录数据出处；"
                   "重训一次后即可自动比对。")

    # 表单内的控件值不会即时回传，故不用 st.form；改用版本号 key 在保存后重置控件
    if "ds_up_ver" not in st.session_state:
        st.session_state.ds_up_ver = 0
    if "ds_lp_ver" not in st.session_state:
        st.session_state.ds_lp_ver = 0
    _uv, _lv = st.session_state.ds_up_ver, st.session_state.ds_lp_ver

    _ups = st.file_uploader(
        "上传 CSV（可多选）", type=["csv"], accept_multiple_files=True,
        key=f"ds_up_{_uv}",
        help=f"单个文件建议 ≤ {_MAX_UP_MB} MB（当前上限）。GB 级数据请用下面的"
             f"「本地路径登记」，避免长时间传输。")
    _up_name = st.text_input("数据集名称（可选）", key=f"ds_upname_{_uv}",
                             placeholder="如：深圳 2021 秋季")
    if st.button("保存上传数据", disabled=not _ups, key=f"ds_upbtn_{_uv}",
                 width="stretch"):
        with st.spinner("保存并校验字段…"):
            _ds, _checks = dreg.add_uploaded(_up_name, _ups)
        _bad = [c for c in _checks if not c["ok"]]
        st.success(f"已登记「{_ds['name']}」：{len(_ds['files'])} 个文件，"
                   f"{len(_checks) - len(_bad)} 个字段校验通过")
        for c in _bad:
            st.error(f"{c['file']}：{c['msg']}")
        st.session_state.ds_up_ver += 1
        st.rerun()

    _lp = st.text_input("登记本机路径（目录或单个 CSV）", key=f"ds_lp_{_lv}",
                        placeholder=r"./data")
    _ln = st.text_input("数据集名称（可选）", key=f"ds_lpname_{_lv}",
                        placeholder="如：深圳 2021 Q4")
    if st.button("登记本地路径", disabled=not _lp.strip(), key=f"ds_lpbtn_{_lv}",
                 width="stretch"):
        try:
            with st.spinner("扫描并校验字段…"):
                _ds, _checks = dreg.add_local(_ln, _lp)
        except Exception as e:
            st.error(f"登记失败：{e}")
        else:
            _bad = [c for c in _checks if not c["ok"]]
            st.success(f"已登记「{_ds['name']}」：{len(_ds['files'])} 个文件"
                       f"（不拷贝，直接引用原路径）")
            for c in _bad:
                st.error(f"{c['file']}：{c['msg']}")
            st.session_state.ds_lp_ver += 1
            st.rerun()

    st.markdown("---")
    for _d in dreg.list_datasets():
        _sz = _gb(sum(f["size"] for f in _d["files"]))
        _lbl = (f"{_d['name']}｜{_d['source']}｜{len(_d['files'])} 文件｜{_sz}")
        _en = st.checkbox(_lbl, value=_d.get("enabled", True),
                          key=f"ds_en_{_d['id']}",
                          help="勾选=参与训练。取消勾选即停用，不移动任何文件。")
        if _en != _d.get("enabled", True):
            dreg.set_enabled(_d["id"], _en)
            st.rerun()
        st.caption(f"登记于 {_d['added_at']}"
                   + (f"｜{_d['note']}" if _d.get("note") else ""))
        _pend = st.session_state.get("ds_del_pending")
        if _pend == _d["id"]:
            st.caption("确认移除？上传的数据集将删除副本；本地登记只解除登记，"
                       "**不会删除你本机的文件**。")
            _c1, _c2 = st.columns(2)
            if _c1.button("确认移除", key=f"ds_ok_{_d['id']}", width="stretch"):
                dreg.remove(_d["id"])
                st.session_state.pop("ds_del_pending", None)
                st.rerun()
            if _c2.button("取消", key=f"ds_no_{_d['id']}", width="stretch"):
                st.session_state.pop("ds_del_pending", None)
                st.rerun()
        elif st.button("移除该数据集", key=f"ds_del_{_d['id']}", width="stretch"):
            st.session_state["ds_del_pending"] = _d["id"]
            st.rerun()

if st.sidebar.button("🔄 重新训练模型", width="stretch",
                      help="用当前启用的数据集重新训练 梯度提升树/Transformer/LSTM/ConvLSTM "
                           "并更新模型文件(含超参搜索与早停,耗时约 10-20 分钟)"):
    with st.status("正在训练模型,请稍候...", expanded=True) as status:
        try:
            r = subprocess.run(
                [sys.executable, "-X", "utf8",
                 os.path.join("src", "period_trainer.py"), "--force"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                cwd=PROJECT_ROOT, timeout=3600,
            )
            ok = (r is not None and r.returncode == 0)
            # 训练日志实时展示到 UI(尾部 1500 字符)
            log_txt = (r.stdout or "")[-1500:] if r else ""
            if log_txt:
                st.code(log_txt)
            detail = (r.stderr or "")[-600:] if r else str(Exception("no result"))
            if ok:
                status.update(label="✅ 训练完成", state="complete")
            else:
                status.update(label="❌ 训练失败", state="error")
        except Exception as e:
            ok = False
            detail = str(e)
            status.update(label="❌ 训练异常", state="error")
    if ok:
        st.sidebar.success("✅ 训练完成,页面刷新中...")
        st.cache_resource.clear()
        st.cache_data.clear()      # 指标/延迟/预测缓存全部随新模型失效
        st.rerun()
    else:
        st.sidebar.error(f"训练失败: {detail}")

st.sidebar.markdown("---")
# 数据描述全部由「启用数据集 + 模型 meta」生成，换/加数据集后自动跟着变
_ds_desc = (f"{_ds_sum['n_active']} 个启用数据集 · {_ds_sum['n_files']} 个文件"
            f" · {_gb(_ds_sum['total_bytes'])}")
if pt_info:
    _ds_desc += f"\n🧠 已训练: {pt_info['data_range'][0]} ~ {pt_info['data_range'][1]}"
    _ds_desc += f"（{pt_info['n_hours']} 小时）"
else:
    _ds_desc += "\n🧠 尚未训练"
st.sidebar.markdown(
    f"📊 训练数据: {_ds_desc}\n"
    f"🔧 网格: {gs}×{gs} (池化 {pg}×{pg})\n"
    + (f"📅 滚动区间: {_dmin} ~ {_dmax}" if rolling and _roll_start is not None
       else "📅 时间轴: 任意未来日期")
)

st.sidebar.markdown("---")

# ═══ 网格点击状态 ═══
if "clicked_grid" not in st.session_state:
    st.session_state.clicked_grid = (10, 10)

# ═══ 核心预测(全局缓存) ═══
# 缓存键含模型文件 mtime: 重新训练后 mtime 变化 → 缓存自动失效,无需手动刷新
@st.cache_data(ttl=60, show_spinner=False)
def _cached_pred(target_date, hour, weather, evt, mkey, model_mtime, hol_ovr,
                 mode="cold", resid=None, wnorm=None, precip=None, win_hours=None):
    """单小时预测（缓存）。

    缓存键同时包含 mode 与滚动上下文（resid / wnorm / precip / win_hours）：
    同一时刻在两种口径下的结果互不覆盖，也不会因上下文全为 None 而与冷启动
    缓存命中同一项。
    """
    models = _cached_models(model_mtime)
    rw = None
    if resid is not None:
        rw = {"norm": wnorm, "precip_mm": precip, "hours": win_hours}
    return predict_hour(str(target_date), hour, weather, event=evt, model_key=mkey,
                        models=models, holiday_override=hol_ovr,
                        recent_residual=resid, rolling_weather=rw,
                        baseline_which="eval" if resid is not None else "deploy")


pred = _cached_pred(date, hour, weather_mode, event, model_key, _model_mtime(), holiday_override,
                    mode, _resid, _wnorm, _precip, _win_hours)
if pred is None:
    st.error("⚠️ 模型文件缺失或未训练。请点击左侧「🔄 重新训练模型(约30秒)」按钮完成训练后重试；若按钮报错,请查看终端日志。")
    st.stop()
pred = np.clip(pred, 0, None)

# ═══ 侧边栏数值监控 ═══
with st.sidebar.expander("📊 当前预测数值", expanded=True):
    st.caption(f"全网总需求({hour:02d}:00): **{int(np.ceil(pred.sum()))} 辆**")
    st.caption(f"全网均值:{pred.mean():.1f} | 峰值网格:{int(np.ceil(pred.max()))} 辆")
    top_idx = np.unravel_index(np.argmax(pred), pred.shape)
    st.caption(f"峰值位置:({top_idx[0]},{top_idx[1]})")

# ═══ 标题区 ═══
ts = pd.Timestamp(f"{date} {hour:02d}:00:00")
dow_str = ["周一","周二","周三","周四","周五","周六","周日"][ts.dayofweek]
hol = holiday_name(ts)
if holiday_override is True:
    hol_hint = "(强制节假日)"
elif holiday_override is False:
    hol_hint = "(强制工作日)"
else:
    hol_hint = f"({hol})" if hol else ("(普通工作日)" if ts.dayofweek < 5 else "(普通周末)")
st.title("🚲 基于时空数据的共享单车需求预测与调度系统")
_mode_tag = "🔁 滚动口径（已接入真实历史）" if rolling else "🔮 冷启动口径（纯未来预测）"
_weather_desc = "天气:历史真实值" if rolling else f"天气:{weather_mode}"
st.caption(
    f"{_mode_tag} | {date} {hour:02d}:00 | {dow_str}{hol_hint} | "
    f"{_weather_desc} | 事件:{event_type} | 模型:{model_choice}"
)

# ═══ 热力地图区域 ═══
st.subheader("📍 需求热力地图（渐变色块 + 点击联动；切换参数后自动更新）")

# 固定色标:使不同天气/时间整体需求差异直接反映在颜色上
# hourly_tensor 统计: mean=155.7, p90=444, p99=2084; 取 1200 作为色标上限
GLOBAL_CMAX = 1200.0

# 读取网格边界
grid_meta_path = os.path.join(config.PROCESSED_DIR, "grid_metadata.pkl")
if not os.path.exists(grid_meta_path):
    st.error("⚠️ 网格元数据不存在,请先运行预处理")
    st.stop()
with open(grid_meta_path, "rb") as f:
    grid_meta = pickle.load(f)
lng_edges = [float(v) for v in grid_meta["lng_edges"]]
lat_edges = [float(v) for v in grid_meta["lat_edges"]]
map_center = [(lat_edges[0] + lat_edges[-1]) / 2, (lng_edges[0] + lng_edges[-1]) / 2]

# ─── 颜色映射函数(绿→黄→红,线性) ───
def _density_color(val, max_val):
    """需求密度 → 绿→黄→红 HTML 颜色字符串"""
    intensity = min(max(val / max_val, 0.0), 1.0)
    if intensity < 0.5:
        t = intensity / 0.5
        r_c, g_c = int(255 * t), 255
    else:
        t = (intensity - 0.5) / 0.5
        r_c, g_c = 255, int(255 * (1 - t))
    return f"#{r_c:02x}{g_c:02x}00"

# ─── folium 地图构建 ───
import folium
from streamlit_folium import st_folium


class _LocalMap(folium.Map):
    # folium 默认从 cdn.jsdelivr.net / code.jquery.com 拉取 leaflet 与 jQuery,
    # 现场断网或 Edge 跟踪防护拦截时整个地图组件无法渲染(热力图消失)。
    # 改为引用本机静态文件(app/static/, 经 Streamlit /app/static/ 路径提供),
    # 地图渲染不再依赖任何外链; 底图瓦片(Esri/高德)仍走在线服务。
    # jQuery/bootstrap/awesome-markers 本项目未用到(无依赖它们的插件), 直接去掉。
    default_js = [("leaflet", "/app/static/leaflet.js")]
    default_css = [("leaflet_css", "/app/static/leaflet.css")]


m = _LocalMap(location=map_center, zoom_start=13, tiles=None)
# 双底图(用户可切换)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri 街道图", max_zoom=19,
).add_to(m)
folium.TileLayer(
    tiles="https://webst01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
    attr="高德", name="高德街道图", max_zoom=19,
).add_to(m)

# 当前选中网格
sx, sy = st.session_state.clicked_grid

# ─── 渐变底图(ImageOverlay):从模型真实分辨率 5×5 直接高倍双线性上采样+高斯模糊 ───
# 说明: 模型在 4×4 求和池化后的 5×5 网格上训练(pg=gs//POOL),
# 预测值经块状复制还原为 20×20 展示。渐变底图直接用 5×5 预测值
# 双线性放大,再用高斯模糊平滑块间硬边(插值/模糊仅为展示平滑,
# 不代表模型空间分辨率高于 5×5)。
if gs <= 40:
    # (pg,pg) 模型真实分辨率 → 归一化 → 双线性放大 → 高斯模糊
    pred_p = np.clip(pred.reshape(pg, POOL, pg, POOL).mean(axis=(1, 3)), 0, None)  # 5×5 区块均值
    norm = np.clip(pred_p / GLOBAL_CMAX, 0.0, 1.0)
    img_small = _PIL_Img.fromarray((norm * 255).astype(np.uint8), mode="L")
    img_big = img_small.resize((600, 600), _PIL_Img.BILINEAR)
    # 高斯模糊: 消除块间硬边(半径≈1.5 个源网格像素, 600/5*1.5≈180 中心区域)
    img_big = img_big.filter(_PIL_ImageFilter.GaussianBlur(radius=140))
    # 注意: 不做对比度拉伸! 直接用 pred_p/GLOBAL_CMAX 绝对值映射,
    # 保证凌晨低需求(小值)呈现绿色而非被拉伸成红色
    arr = np.asarray(img_big, dtype=np.float64) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    # 绿→黄→红(与 _density_color 相同的线性映射)
    r_ch = np.where(arr < 0.5, arr / 0.5 * 255.0, 255.0)
    g_ch = np.where(arr < 0.5, 255.0, 255.0 * (1.0 - (arr - 0.5) / 0.5))
    b_ch = np.zeros_like(arr)
    rgb_arr = np.stack([r_ch, g_ch, b_ch], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    _PIL_Img.fromarray(rgb_arr, "RGB").save(buf, format="PNG")
    b64s = base64.b64encode(buf.getvalue()).decode()
    folium.raster_layers.ImageOverlay(
        image="data:image/png;base64," + b64s,
        bounds=[[lat_edges[0], lng_edges[0]], [lat_edges[-1], lng_edges[-1]]],
        opacity=0.6,
        interactive=False,  # 禁止拦截鼠标事件，让 Rectangle 交互层接收点击
        className="gradient-overlay",
        name="需求渐变底图",
    ).add_to(m)
# 关键: 让渐变底图 img 彻底不接收命中测试(pointer-events:none),
# 否则点击命中 img 后无法穿透到下方 Rectangle 交互层(Leaflet 事件代理限制)
folium.Element(
    '<style>.gradient-overlay{pointer-events:none !important}</style>'
).add_to(m)

# ─── 交互网格层:仅淡边框+tooltip+点击(颜色视觉交给渐变底图) ───
# 放入 FeatureGroup 并传 feature_group_to_add, 触发 st_folium 前端 onLayerClick 事件绑定
interactive_fg = folium.FeatureGroup(name="交互网格")
for x in range(gs):
    for y in range(gs):
        val = float(pred[x, y])
        is_sel = (x == sx and y == sy)
        tip = f"({x},{y}): {int(np.ceil(val))} 辆"
        folium.Rectangle(
            bounds=[[lat_edges[y], lng_edges[x]], [lat_edges[y + 1], lng_edges[x + 1]]],
            color="#ffffff" if is_sel else "#ffffff",
            weight=3.0 if is_sel else 0.15,
            fill=True,  # SVG 命中测试要求 fill 非 none; opacity=0 不影响命中但视觉透明
            fill_color="#ffffff",
            fill_opacity=0.0,
            tooltip=tip,
        ).add_to(interactive_fg)
# 注意: 这里**不要** interactive_fg.add_to(m)。
# folium 的 Layer.render() 每次渲染都会往要素组里追加一条 "<组变量名>.addTo(map)" 语句，
# 而 streamlit_folium 为 feature_group_to_add 另起一套 JS 变量名(形如 feature_group_feature_group_0)。
# 若先把要素组加进地图，地图脚本里那条 addTo 用的是渲染前的随机变量名(uuid)，
# 在 iframe 里从未定义 —— 浏览器抛 "feature_group_xxxx is not defined"，
# 且该错误发生在要素组脚本中段，后面的 window.feature_group.push(...) 不会执行，
# 点击网格回传会失效。交给 st_folium 单独挂载即可，同时也避免 400 个矩形被渲染两遍。
# ─── 事件扰动标注图层: 让事件中心与影响范围在地图上可见 ───
# 背景: 事件影响区若色块值已超色标上限,热力图颜色不变,用户感知不到事件生效。
# 方案: 在事件中心画圆圈(活动=绿,事故=红),标注影响范围半径。
if event is not None:
    ev_lat = lat_edges[event["y"]] + (lat_edges[event["y"] + 1] - lat_edges[event["y"]]) / 2
    ev_lng = lng_edges[event["x"]] + (lng_edges[event["x"] + 1] - lng_edges[event["x"]]) / 2
    ev_r = event.get("radius", 2)
    # 影响范围圆圈: 半径= radius 格 × 单格尺寸
    cell_lat = (lat_edges[1] - lat_edges[0])
    cell_lng = (lng_edges[1] - lng_edges[0])
    ev_color = "#2ecc71" if event["type"] == "activity" else "#e74c3c"
    ev_label = "活动(需求↑)" if event["type"] == "activity" else "事故(需求↓)"
    ev_sign = "+" if event["type"] == "activity" else "-"
    folium.CircleMarker(
        location=[ev_lat, ev_lng],
        radius=10,
        color=ev_color,
        weight=3,
        fill=True,
        fill_color=ev_color,
        fill_opacity=0.4,
        tooltip=(f"{ev_label} 中心 ({event['x']},{event['y']}) "
                 f"强度{ev_sign}{int(event.get('strength', 0.3) * 100)}% 半径{ev_r}格"
                 f" — 点击地图任一网格可更改中心"),
        name="事件标注",
    ).add_to(m)
    folium.Circle(
        location=[ev_lat, ev_lng],
        radius=max(ev_r * cell_lat * 111320.0, 200),  # 度→米近似(纬度1°≈111.32km)
        color=ev_color,
        weight=1.5,
        dashArray="4,4",
        fill=False,
        tooltip=f"事件影响范围 (半径{ev_r}格)",
        name="事件影响范围",
    ).add_to(m)

# LayerControl 必须在所有图层之后(folium 0.20 setZIndex bug)
folium.LayerControl().add_to(m)
m.fit_bounds([[lat_edges[0], lng_edges[0]], [lat_edges[-1], lng_edges[-1]]])

# ─── st_folium 组件(捕获点击事件→更新选中网格) ───
# feature_group_to_add 触发前端遍历 FG layers 并绑定 onLayerClick 处理器,
# 否则 Rectangle 直接 add_to(m) 时前端无法捕获 click 事件
click_data = st_folium(
    m,
    width=970,
    height=560,
    key="demand_map",
    returned_objects=["last_object_clicked_tooltip", "last_clicked"],
    feature_group_to_add=interactive_fg,
)
tip_clicked = (click_data or {}).get("last_object_clicked_tooltip")
if tip_clicked:
    mm = re.search(r"\((\d+),(\d+)\)", str(tip_clicked))
    if mm:
        # 修复(2026-10-13): 选中网格 sx,sy 在本行之前的渲染中已被读取(见"当前选中网格")，
        # 原先只写 session_state 会让界面比实际点击慢一步——点第 N 格要等下一次交互才刷新。
        # 现改为值真正变化时写回并立即 rerun；同值不重复 rerun，避免自触发死循环。
        _new_grid = (int(mm.group(1)), int(mm.group(2)))
        if _new_grid != st.session_state.clicked_grid:
            st.session_state.clicked_grid = _new_grid
            # 事件中心跟随点击:点哪一格,事件就发生在哪一格。
            # 递增 ev_ver 使侧边栏输入框以新 key 重建并读取新值
            if event_type != "无":
                st.session_state.ev_center = _new_grid
                st.session_state.ev_ver = st.session_state.ev_ver + 1
            st.rerun()

# ─── 色标图例 ───
legend_html = (
    '<div style="display:flex;align-items:center;gap:5px;font-size:12px;color:#444">'
    '<span>0 辆</span>'
    '<div style="width:180px;height:13px;border-radius:3px;'
    'background:linear-gradient(to right,#00ff00,#ffff00,#ff0000)"></div>'
    '<span>≥1200 辆/网格/时</span>'
    '</div>'
)
st.markdown(legend_html, unsafe_allow_html=True)

# 点击提示
st.caption(
    f"🎯 点击地图网格可联动曲线"
    + ("，并**将该格设为事件中心**" if event_type != "无" else "")
    + f" | 当前选中:**({sx},{sy})** "
    f"当前值:**{int(np.ceil(pred[sx, sy]))} 辆** "
    f"| 图层控制可切换底图"
)

# 滚动口径状态提示：明确告知用户"读到的是哪一段真实数据"
if rolling and roll_ctx is not None:
    st.info(
        f"🔁 **滚动口径**：已自动读取 {roll_ctx['start']:%m-%d %H:%M} ~ "
        f"{roll_ctx['end']:%m-%d %H:%M} 共 {roll_ctx['seq_len']} 小时的真实需求残差，"
        f"外生特征与基线降雨档同步取历史真实值；"
        f"该结果与「模型信息」中的 rolling 指标同口径，可直接对照。"
    )
elif rolling:
    st.info(
        f"🔁 **滚动口径**：{model_choice} 无序列结构（没有历史残差通道），"
        f"此处结果与冷启动完全一致。切换到 LSTM / ConvLSTM 才能看到滚动增益。"
    )

# ═══ Tab 区 ═══
st.markdown("---")
tab_disp, tab_dispatch = st.tabs(["📈 24小时需求曲线", "🚚 调度建议"])

# ─── Tab1: 24小时需求曲线 ───
with tab_disp:
    day_pred = predict_day(str(date), weather_mode, event=event, model_key=model_key,
                           holiday_override=holiday_override, models=_models_obj,
                           rolling=rolling)
    if day_pred is None:
        st.warning("请先训练模型")
    else:
        # 选中网格曲线
        curve_vals = [int(np.ceil(day_pred[h, sx, sy])) for h in range(24)]
        max_h = max(curve_vals) if curve_vals else 0
        st.caption(
            f"**网格 ({sx},{sy}) 24小时需求曲线** | "
            f"{date} {_weather_desc} | 峰值 {max_h} 辆 | "
            + ("逐小时各取一段历史窗口 | " if rolling else "")
            + "点击地图网格可切换"
        )
        chart_sel = pd.DataFrame({"小时": list(range(24)), "预测需求": curve_vals})
        st.line_chart(chart_sel, x="小时", y="预测需求", height=280)

        st.markdown("**全网总需求（24小时）**")
        total_vals = [int(np.ceil(day_pred[h].sum())) for h in range(24)]
        chart_total = pd.DataFrame({"小时": list(range(24)), "全网总需求": total_vals})
        st.line_chart(chart_total, x="小时", y="全网总需求", height=260)

# ─── Tab2: 调度建议 ───
with tab_dispatch:
    st.caption("基于全天 24 小时预测的网格调度建议（不依赖实时库存，基于预测需求趋势）")
    st.caption("**⚙️ 说明**：模型空间分辨率为 5×5 共 25 个区块（20×20 网格做 4×4 求和池化后训练），")
    st.caption("同一区块内 16 个网格共享同一预测值。调度建议按**区块**给出，代表网格=区块中心格。")
    st.caption("**投放建议**：预测需求持续高 → 优先在高需求区块投放车辆")
    st.caption("**回收建议**：预测需求持续低 → 从低需求区块回收车辆至需求高的区块")

    # 区块级聚合 (pg×pg=25): 每区块取中心网格为代表
    pg = gs // POOL
    rep = {(bx, by): (bx * POOL + POOL // 2, by * POOL + POOL // 2)
           for bx in range(pg) for by in range(pg)}
    block_totals = {(bx, by): day_pred[:, bx * POOL + POOL // 2, by * POOL + POOL // 2].sum()
                    for bx in range(pg) for by in range(pg)}
    blocks_sorted = sorted(block_totals.items(), key=lambda kv: kv[1], reverse=True)

    # 投放/回收各取 10 个区块，保证两份清单互不重叠（原为 20+10 > 25 会重复 5 个区块）；
    # 其余区块维持现状，不做调度动作
    top_nb = min(10, pg * pg // 2)
    bottom_nb = min(10, pg * pg // 2)
    st.caption(
        f"**调度范围**：{pg * pg} 个区块中需求最高的 {top_nb} 个投放、最低的 {bottom_nb} 个回收，"
        f"其余 {pg * pg - top_nb - bottom_nb} 个区块维持现状。"
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"### 🔴 投放建议（需求最高的 {top_nb} 个区块，排名 1 = 最该投放）")
        rows = []
        for rank, ((bx, by), tot) in enumerate(blocks_sorted[:top_nb], 1):
            cx, cy = rep[(bx, by)]
            peak_h = int(np.argmax(day_pred[:, cx, cy]))
            peak_v = int(np.ceil(day_pred[peak_h, cx, cy]))
            rows.append({
                "排名": rank,
                "区块": f"({bx},{by})",
                "中心网格": f"({cx},{cy})",
                "全天总需求": int(np.ceil(tot)),
                "峰值时刻": f"{peak_h:02d}:00 ({peak_v}辆)",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={c: st.column_config.Column(alignment="center")
                                    for c in df.columns})

    with col_r:
        st.markdown(f"### 🔵 回收建议（需求最低的 {bottom_nb} 个区块，排名 1 = 最该回收）")
        rows2 = []
        # 升序排列：需求最低的区块排第 1，与“排名越高越紧急”的直觉一致
        for rank, ((bx, by), tot) in enumerate(blocks_sorted[-bottom_nb:][::-1], 1):
            cx, cy = rep[(bx, by)]
            trough_h = int(np.argmin(day_pred[:, cx, cy]))
            trough_v = int(np.ceil(day_pred[trough_h, cx, cy]))
            rows2.append({
                "排名": rank,
                "区块": f"({bx},{by})",
                "中心网格": f"({cx},{cy})",
                "全天总需求": int(np.ceil(tot)),
                "低谷时刻": f"{trough_h:02d}:00 ({trough_v}辆)",
            })
        df2 = pd.DataFrame(rows2)
        st.dataframe(df2, width="stretch", hide_index=True,
                     column_config={c: st.column_config.Column(alignment="center")
                                    for c in df2.columns})

    st.markdown("### 📊 全区块需求概览（按全天总需求降序）")
    all_rows = []
    for (bx, by), tot in blocks_sorted:
        cx, cy = rep[(bx, by)]
        all_rows.append({
            "区块": f"({bx},{by})",
            "中心网格": f"({cx},{cy})",
            "全天总需求": int(np.ceil(tot)),
            "小时均需求": f"{tot / 24:.1f}",
            "峰值": int(np.ceil(day_pred[:, cx, cy].max())),
            "低谷": int(np.ceil(day_pred[:, cx, cy].min()))})
    df_all = pd.DataFrame(all_rows)
    st.dataframe(df_all, width="stretch", hide_index=True, height=420,
                 column_config={c: st.column_config.Column(alignment="center")
                                for c in df_all.columns})

    st.info(
        "💡 调度策略：优先从低需求区块回收车辆，投放至高需求区块。"
        "模型分辨率 5×5 区块，同一区块内预测值一致（求和池化还原）。"
        "建议结合实时库存数据动态调整，实际运营中考虑车辆调度成本。"
    )
