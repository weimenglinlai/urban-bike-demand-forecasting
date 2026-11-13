"""
日期与气象特征模块（Date & Weather Features）
================================================
把「这一天在这一小时是否会发生出行需求变化」转化为模型可学习的数值特征。

两条设计主线
------------
1. **外生特征 z（未来可知）**：小时、星期、周末/法定假日/调休上班日、假期长度与阶段、
   气温/降水/风速/云量。z 只依赖日历与气象预报，**不含任何需求历史**，
   因此对未来任意日期都可构造，是全系统「纯未来预测」口径的前提。

2. **气象情景的逐小时廓线**：早期实现把气象情景（晴天/大雨…）在预测时
   铺成 24 小时常数，而训练时用的是逐小时真实变化的气象序列，
   造成「训练—推理分布不一致」。本模块改为从训练集真实气象中按情景
   抽取样本、统计**典型日廓线**（24 小时各不相同），预测时按廓线展开，
   使训练与推理的输入分布一致。

日期类型（day_type）分四类，供条件基线分组使用：
    0 普通工作日 / 1 普通周末 / 2 法定节假日 / 3 调休上班日

用法:
    from src.date_features import make_features, FEATURE_NAMES, day_type, rain_bin
    z = make_features(pd.date_range("2021-07-01", periods=24, freq="h"), scenario="晴天")
"""

import os
import numpy as np
import pandas as pd

try:
    import chinese_calendar as cc

    _HAS_CC = True
except ImportError:
    _HAS_CC = False

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEATHER_DIR = os.path.join(_BASE_DIR, "data", "processed", "weather")

# 公历固定节日（每年相同），用于 chinese_calendar 覆盖范围之外的年份
_FIXED_HOLIDAYS = [
    ("01-01", "元旦"),
    ("05-01", "劳动节"),
    ("06-01", "儿童节"),
    ("10-01", "国庆节"),
    ("12-25", "圣诞节"),
]

# ---------------------------------------------------------------------------
# 外生特征定义（z）：13 维，全部「未来可知」
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_weekend", "is_holiday", "is_makeup",
    "holiday_len", "holiday_day",
    "temp", "precip", "wind", "cloud",
]
N_FEATURES = len(FEATURE_NAMES)

# 日期类型（条件基线分组维度）
DAY_TYPE_NAMES = ["工作日", "周末", "法定假日", "调休上班日"]
N_DAY_TYPES = len(DAY_TYPE_NAMES)

# 降雨分档（气象业务惯例，单位 mm/h）
RAIN_EDGES = (0.1, 2.5, 8.0)
RAIN_LABELS = ["无雨", "小雨", "中雨", "大雨"]
N_RAIN_BINS = len(RAIN_LABELS)

# 气象情景
WEATHER_SCENARIOS = ["晴天", "阴天", "小雨", "大雨", "高温"]

# 情景判定规则（作用于原始气象量，用于从历史中抽取该情景的样本）
SCENARIO_RULES = {
    "晴天": dict(precip_max=0.1, cloud_max=40.0),
    "阴天": dict(precip_max=0.1, cloud_min=40.0),
    "小雨": dict(precip_min=0.1, precip_max=2.5),
    "大雨": dict(precip_min=2.5),
    "高温": dict(temp_pct=90.0),
}

# 情景廓线的兜底形状统一由 _default_profiles() 内嵌维护（见下方），
# 不再在此处散落顶层常量，避免「常量」与「函数」两处各写一份导致不一致。


# ---------------------------------------------------------------------------
# 节假日与日期类型
# ---------------------------------------------------------------------------

def _holiday_detail(date: pd.Timestamp):
    """返回 (是否法定节假日, 是否调休上班日, 假日名称)。

    依据 chinese_calendar.get_holiday_detail 的语义：
      - (True,  名称)   → 法定节假日（含假期内每一天）
      - (False, 名称)   → 调休上班日（周末被调整为上班，名称沿用所属假期）
      - (True,  None)   → 普通周末（休息但不是法定节假日）
      - (False, None)   → 普通工作日
    """
    try:
        if _HAS_CC:
            detail = cc.get_holiday_detail(date.date())
            if isinstance(detail, (tuple, list)) and len(detail) >= 2:
                is_off, name = bool(detail[0]), detail[1]
                has_name = isinstance(name, str) and bool(name.strip())
                if is_off and has_name:
                    return True, False, name
                if (not is_off) and has_name:
                    return False, True, name
                return False, False, ""
    except (NotImplementedError, ValueError):
        pass
    # 覆盖范围之外：按公历固定节日推断
    mmdd = date.strftime("%m-%d")
    for f, n in _FIXED_HOLIDAYS:
        if mmdd == f:
            return True, False, n
    return False, False, ""


def is_holiday(date: pd.Timestamp) -> bool:
    """是否为法定节假日（普通周末不算）。"""
    return _holiday_detail(date)[0]


def is_makeup_workday(date: pd.Timestamp) -> bool:
    """是否为调休上班日（周末被调为上班）。

    这类日期的需求模式更接近工作日而非周末，早期版本把它与普通工作日
    完全等同，导致「假期与普通日」的刻画不完整。
    """
    return _holiday_detail(date)[1]


# 英文节假日名 → 中文映射（chinese_calendar 返回英文）
_HOLIDAY_NAME_CN = {
    "New Year's Day": "元旦",
    "Spring Festival": "春节",
    "Tomb-Sweeping Day": "清明节",
    "Tomb-sweeping Day": "清明节",
    "Labour Day": "劳动节",
    "Labor Day": "劳动节",
    "Dragon Boat Festival": "端午节",
    "Mid-autumn Festival": "中秋节",
    "Mid-Autumn Festival": "中秋节",
    "National Day": "国庆节",
}


def holiday_name(date: pd.Timestamp) -> str:
    """节假日名称（尽力而为；未知返回空串）。"""
    _, _, name = _holiday_detail(date)
    if not name:
        return ""
    return _HOLIDAY_NAME_CN.get(name, name)


def is_workday(date: pd.Timestamp) -> bool:
    """是否为工作日（非休假）。

    修正说明：早期实现写作 ``return not is_holiday(date)``，会把普通周末
    误判为工作日。正确语义是「既不是法定节假日、也不是周末（调休上班日除外）」。
    """
    hol, makeup, _ = _holiday_detail(date)
    if hol:
        return False
    if date.dayofweek >= 5 and not makeup:
        return False
    return True


def day_type(date: pd.Timestamp) -> int:
    """日期类型：0 工作日 / 1 周末 / 2 法定假日 / 3 调休上班日。"""
    hol, makeup, _ = _holiday_detail(date)
    if hol:
        return 2
    if makeup:
        return 3
    return 1 if date.dayofweek >= 5 else 0


def day_type_array(hours) -> np.ndarray:
    ts = pd.to_datetime(hours)
    return np.array([day_type(t) for t in ts], dtype=np.int64)


def holiday_span(date: pd.Timestamp):
    """返回 (假期总天数, 假期内第几天)；非假期返回 (0, 0)。

    用于刻画「假期首日 / 中段 / 末日」的需求差异——例如国庆首日出城、
    末两日返程的模式与假期中段明显不同。
    """
    if not is_holiday(date):
        return 0, 0
    lo = date
    while is_holiday(lo - pd.Timedelta(days=1)):
        lo -= pd.Timedelta(days=1)
        if (date - lo).days > 30:  # 防御：极端异常时终止
            break
    hi = date
    while is_holiday(hi + pd.Timedelta(days=1)):
        hi += pd.Timedelta(days=1)
        if (hi - date).days > 30:
            break
    return (hi - lo).days + 1, (date - lo).days + 1


def holiday_fields(hours) -> np.ndarray:
    """批量返回 (n, 2) = [假期总天数, 假期内第几天]。"""
    ts = pd.to_datetime(hours)
    cache = {}
    out = np.zeros((len(ts), 2), dtype=np.float64)
    for i, t in enumerate(ts):
        key = t.normalize()
        if key not in cache:
            cache[key] = holiday_span(key)
        ln, pos = cache[key]
        out[i, 0] = ln
        out[i, 1] = pos
    return out


def rain_bin(precip) -> np.ndarray:
    """降水（mm/h，原始量）→ 分档索引 0 无雨 / 1 小雨 / 2 中雨 / 3 大雨。"""
    p = np.asarray(precip, dtype=np.float64)
    return np.digitize(p, RAIN_EDGES, right=False).astype(np.int64)


# ---------------------------------------------------------------------------
# 气象情景：训练集典型日廓线
# ---------------------------------------------------------------------------

def _scenario_sample_mask(scenario: str, weather_raw: np.ndarray):
    """从真实气象序列中挑出属于某情景的小时（仅用训练集统计）。"""
    rule = SCENARIO_RULES.get(scenario, {})
    temp, precip, wind, cloud = (weather_raw[:, i] for i in range(4))
    m = np.ones(len(weather_raw), dtype=bool)
    if "precip_min" in rule:
        m &= precip > rule["precip_min"]
    if "precip_max" in rule:
        m &= precip <= rule["precip_max"]
    if "cloud_min" in rule:
        m &= cloud >= rule["cloud_min"]
    if "cloud_max" in rule:
        m &= cloud <= rule["cloud_max"]
    if "temp_pct" in rule:
        m &= temp >= np.percentile(temp, rule["temp_pct"])
    return m


def build_scenario_profiles(hours, weather_raw: np.ndarray, train_mask=None) -> dict:
    """按「情景 × 小时」统计真实气象的典型日廓线（RAW 单位）。

    返回 dict[scenario] -> (24, 4) 数组；样本不足的情景回退到「训练集逐小时
    均值廓线 + 情景化降水缩放」，保证廓线物理合理、逐小时变化。

    必须在**训练集小时**上统计（train_mask），避免把测试段气象统计量带入。
    """
    ts = pd.to_datetime(hours)
    hod = ts.hour.values
    w = np.asarray(weather_raw, dtype=np.float64)
    if train_mask is None:
        train_mask = np.ones(len(w), dtype=bool)
    train_mask = np.asarray(train_mask, dtype=bool)

    # 训练集逐小时整体均值廓线（所有情景的公共形状）
    base = np.zeros((24, 4), dtype=np.float64)
    for h in range(24):
        sel = train_mask & (hod == h)
        if sel.sum() >= 5:
            base[h] = w[sel].mean(axis=0)
        else:
            sel2 = hod == h
            base[h] = w[sel2].mean(axis=0) if sel2.sum() else w.mean(axis=0)

    profiles = {}
    for sc in WEATHER_SCENARIOS:
        m = _scenario_sample_mask(sc, w) & train_mask
        prof = np.zeros((24, 4), dtype=np.float64)
        n_used = 0
        for h in range(24):
            sel = m & (hod == h)
            if sel.sum() >= 3:  # 该情景该小时样本足够 → 用真实均值
                prof[h] = w[sel].mean(axis=0)
                n_used += int(sel.sum())
            else:
                prof[h] = base[h]
        if n_used < 48:  # 该情景整体样本不足（如大雨）→ 用公共廓线 + 情景降水缩放
            prof = base.copy()
            prof[:, 1] = prof[:, 1] * _scenario_precip_scale(sc, w[train_mask][:, 1])
        profiles[sc] = prof.astype(np.float32)
    return profiles


def _scenario_precip_scale(scenario: str, precip_train: np.ndarray) -> float:
    """样本不足情景的降水缩放系数（相对训练集均值）。"""
    mean_p = float(np.mean(precip_train)) if len(precip_train) else 0.3
    target = {
        "晴天": 0.0, "阴天": 0.0, "小雨": 1.0, "大雨": 8.0, "高温": 0.0,
    }.get(scenario, 0.0)
    return float(target / max(mean_p, 1e-6))


def save_scenario_profiles(profiles: dict, path: str = None):
    path = path or os.path.join(_WEATHER_DIR, "scenario_profiles.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: np.asarray(v, dtype=np.float32) for k, v in profiles.items()})
    return path


def load_scenario_profiles(path: str = None):
    path = path or os.path.join(_WEATHER_DIR, "scenario_profiles.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def _default_profiles() -> dict:
    """唯一兜底廓线来源（物理量级合理，不作为主路径）。

    基线形状内嵌于此：深圳 4—7 月气温日较差小、午后多雨、风速日变化平缓，
    各情景在此基础上做温度/降水/云量偏移。新增情景只需扩 WEATHER_SCENARIOS
    与下方三个 dict，不会再出现「顶层常量与函数各写一份」的不一致。
    """
    base_temp = np.array([
        24.0, 23.7, 23.5, 23.3, 23.3, 23.6, 24.4, 25.6, 26.9, 28.1, 29.0, 29.7,
        30.1, 30.2, 29.9, 29.3, 28.5, 27.7, 27.0, 26.4, 25.9, 25.4, 24.9, 24.4,
    ])
    base_wind = np.array([
        10.5, 10.2, 10.0, 9.8, 9.8, 10.0, 10.8, 11.8, 12.6, 13.2, 13.6, 13.9,
        14.1, 14.2, 14.0, 13.5, 12.8, 12.1, 11.6, 11.2, 11.0, 10.8, 10.7, 10.6,
    ])
    base_cloud = np.array([
        72.0, 74.0, 76.0, 77.0, 77.0, 75.0, 72.0, 68.0, 65.0, 63.0, 62.0, 62.0,
        64.0, 68.0, 73.0, 77.0, 79.0, 80.0, 79.0, 77.0, 75.0, 74.0, 73.0, 72.0,
    ])
    precip_level = {"晴天": 0.0, "阴天": 0.0, "小雨": 1.6, "大雨": 8.0, "高温": 0.0}
    dtemp = {"晴天": 2.5, "阴天": -0.5, "小雨": -1.5, "大雨": -3.0, "高温": 4.5}
    dcloud = {"晴天": -30.0, "阴天": 5.0, "小雨": 12.0, "大雨": 22.0, "高温": -40.0}

    out = {}
    for sc in WEATHER_SCENARIOS:
        prof = np.zeros((24, 4), dtype=np.float64)
        prof[:, 0] = base_temp + dtemp.get(sc, 0.0)
        prof[:, 1] = precip_level.get(sc, 0.0)
        prof[:, 2] = base_wind
        prof[:, 3] = np.clip(base_cloud + dcloud.get(sc, 0.0), 0.0, 100.0)
        out[sc] = prof
    return out


def _weather_norm_params():
    """训练集气象归一化参数（由 period_trainer 在训练集上统计并落盘）。"""
    mp = os.path.join(_WEATHER_DIR, "weather_mean.npy")
    sp = os.path.join(_WEATHER_DIR, "weather_std.npy")
    if os.path.exists(mp) and os.path.exists(sp):
        return np.load(mp), np.load(sp)
    return (np.array([26.93, 0.29, 12.03, 69.81]),
            np.array([3.06, 1.11, 5.18, 33.45]))


def scenario_hourly_weather(scenario: str, hours, profiles=None) -> np.ndarray:
    """情景 → 逐小时气象（RAW 单位），形状 (n, 4)。

    与训练时一致：序列在 24 小时内**逐小时变化**（按典型日廓线展开），
    而不是全天铺同一个常数——后者是早期版本训练—推理分布不一致的根因。
    """
    ts = pd.to_datetime(hours)
    if profiles is None:
        profiles = load_scenario_profiles()
    if profiles is None or scenario not in profiles:
        profiles = profiles or _default_profiles()
    prof = np.asarray(profiles.get(scenario, _default_profiles()["晴天"]), dtype=np.float64)
    return prof[ts.hour.values].astype(np.float64)


def scenario_weather_norm(scenario: str, hours, profiles=None) -> np.ndarray:
    """情景 → 逐小时**归一化**气象 (n, 4)，归一化参数为训练集统计量。"""
    raw = scenario_hourly_weather(scenario, hours, profiles)
    wm, ws = _weather_norm_params()
    return (raw - wm) / ws


# ---------------------------------------------------------------------------
# 特征矩阵
# ---------------------------------------------------------------------------

def exogenous_features(hours, weather_norm=None, scenario=None,
                       holiday_override=None, profiles=None) -> np.ndarray:
    """构造外生特征 z，形状 (n, 13)。"""
    ts = pd.to_datetime(hours)
    n = len(ts)
    hour = ts.hour.values.astype(np.float64)
    dow = ts.dayofweek.values.astype(np.float64)

    dates = ts.normalize()
    if holiday_override is None:
        dt = np.array([day_type(t) for t in dates], dtype=np.int64)
    else:
        # 手动覆盖：整体强制为「法定假日」或「工作日」
        dt = np.full(n, 2 if holiday_override else 0, dtype=np.int64)
    is_hol = (dt == 2).astype(np.float64)
    is_mkup = (dt == 3).astype(np.float64)
    is_wknd = (dow >= 5).astype(np.float64) if holiday_override is None else \
        ((dt == 1).astype(np.float64))

    if holiday_override is None:
        hf = holiday_fields(dates)
    else:
        hf = np.zeros((n, 2), dtype=np.float64)
        if holiday_override:
            hf[:, 0], hf[:, 1] = 1.0, 1.0

    if weather_norm is not None:
        w = np.asarray(weather_norm, dtype=np.float64).reshape(n, 4)
    else:
        w = scenario_weather_norm(scenario or "晴天", ts, profiles)

    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        is_wknd, is_hol, is_mkup,
        hf[:, 0] / 8.0, hf[:, 1] / 8.0,
        w,
    ]).astype(np.float32)


def make_features(hours, weather_arr=None, scenario=None, holiday_override=None,
                  profiles=None):
    """向后兼容入口：weather_arr 传入时为已归一化气象（训练用），
    否则按 scenario 的典型日廓线展开（预测用）。"""
    return exogenous_features(hours, weather_norm=weather_arr, scenario=scenario,
                              holiday_override=holiday_override, profiles=profiles)


if __name__ == "__main__":
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8")
    for d in ["2021-04-25", "2021-05-01", "2021-05-03", "2021-05-08",
              "2021-06-12", "2021-07-03", "2021-07-05", "2026-08-14", "2080-06-01"]:
        t = pd.Timestamp(d)
        wk = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][t.dayofweek]
        print(f"{d} {wk} day_type={DAY_TYPE_NAMES[day_type(t)]} "
              f"name={holiday_name(t)!r} span={holiday_span(t)} workday={is_workday(t)}")
    feats = make_features(pd.date_range("2021-07-01", periods=24, freq="h"), scenario="晴天")
    print("特征矩阵:", feats.shape, "降水通道:", np.round(feats[:12, 11], 3).tolist())
    feats2 = make_features(pd.date_range("2021-07-01", periods=24, freq="h"), scenario="大雨")
    print("大雨降水通道:", np.round(feats2[:12, 11], 3).tolist())
