"""
数据加载器 - 统一入口，自动判断真实/虚拟数据
"""

import os
import sys
import pandas as pd

_1GB = 1024 * 1024 * 1024

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import log

def _detect_encoding(path):
    """检测 CSV 文件编码"""
    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']:
        try:
            with open(path, 'r', encoding=enc) as f:
                f.read(1024)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'utf-8'


FIELD_MAP = {
    # ===== CitiBike 2018 =====
    "starttime": "start_time",
    "stoptime": "end_time",
    "start station longitude": "start_lng",
    "start station latitude": "start_lat",
    "end station longitude": "end_lng",
    "end station latitude": "end_lat",
    "tripduration": "tripduration",
    "usertype": "user_type",
    "birth year": "birth_year",
    "gender": "gender",
    # ===== 上海摩拜 =====
    "start_location_x": "start_lng",
    "start_location_y": "start_lat",
    "end_location_x": "end_lng",
    "end_location_y": "end_lat",
    "start_time": "start_time",
    "end_time": "end_time",
    # ===== 通用中文别名 =====
    "开始时间": "start_time",
    "开始经度": "start_lng",
    "开始纬度": "start_lat",
    "结束经度": "end_lng",
    "结束纬度": "end_lat",
    "结束时间": "end_time",
    # ===== 通用英文别名 =====
    "time": "start_time",
    "start_datetime": "start_time",
    "order_time": "start_time",
    "lng": "start_lng",
    "start_longitude": "start_lng",
    "lat": "start_lat",
    "start_latitude": "start_lat",
    "end_longitude": "end_lng",
    "end_latitude": "end_lat",
    # ===== 深圳共享单车 2021 =====
    "START_TIME": "start_time",
    "START_LNG": "start_lng",
    "START_LAT": "start_lat",
    "END_TIME": "end_time",
    "END_LNG": "end_lng",
    "END_LAT": "end_lat",
}


def load_data():
    """加载数据，自动判断来源，返回统一 DataFrame

    文件来源由数据集登记表（src/dataset_registry）决定：只有「启用」的
    数据集参与训练。登记表为空时自动把 data/raw 下的 CSV 登记为内置数据集，
    行为与旧版一致。
    """
    from src.dataset_registry import active_files, ensure_default, summarize

    ensure_default()
    csv_files = active_files()
    if not csv_files:
        raise FileNotFoundError(
            "没有启用的训练数据。请在界面「📁 训练数据」中上传文件或登记本地路径，"
            f"或直接把 CSV 放入 {config.RAW_DIR}"
        )
    log(f"使用真实数据（{summarize()['n_active']} 个启用数据集，{len(csv_files)} 个文件）")
    dfs = []
    first_cols = None
    for f in csv_files:
        path = f                      # 登记表给出的是绝对路径，不再与 RAW_DIR 拼接
        log(f"  加载: {os.path.basename(path)}")
        df_loaded = _load_csv(path)
        if first_cols is None:
            first_cols = set(df_loaded.columns)
        else:
            loaded_cols = set(df_loaded.columns)
            if loaded_cols != first_cols:
                missing = first_cols - loaded_cols
                extra = loaded_cols - first_cols
                msg = []
                if missing:
                    msg.append(f"缺少字段: {missing}")
                if extra:
                    msg.append(f"多余字段: {extra}")
                log(f"  警告: {f} 字段不一致 — {'; '.join(msg)}。已自动对齐。")
                df_loaded = df_loaded.reindex(columns=list(first_cols))
        dfs.append(df_loaded)
    result = pd.concat(dfs, ignore_index=True)
    log(f"  合并完成: {len(csv_files)} 个文件, 共 {len(result):,} 条记录")
    return result


def _load_csv(path: str) -> pd.DataFrame:
    """加载单个 CSV 并标准化字段名"""

    encoding_tries = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]
    df = None
    for enc in encoding_tries:
        try:
            # 先读首行确认列名
            header = pd.read_csv(path, encoding=enc, nrows=0)
            # 判断是否为大文件（>500MB）且无行数限制
            file_size = os.path.getsize(path)
            if file_size > 500 * 1024 * 1024 and config.MAX_ROWS is None:
                log(f"  大文件 ({file_size/_1GB:.1f} GB)，使用 chunk 加载…")
                chunks = []
                for chunk in pd.read_csv(path, encoding=enc,
                                         chunksize=config.ORDER_CHUNK_SIZE):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(path, encoding=enc, nrows=config.MAX_ROWS)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise ValueError(f"无法解码文件: {path}")

    # 字段映射
    rename = {}
    for col in df.columns:
        std = FIELD_MAP.get(col, None)
        if std:
            rename[col] = std
    if rename:
        df.rename(columns=rename, inplace=True)

    required = ["start_time", "start_lng", "start_lat", "end_lng", "end_lat", "end_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        unmatched = [c for c in df.columns if c not in FIELD_MAP]
        hint = ""
        if unmatched:
            hint = (
                f"\n\n未映射的字段: {unmatched}"
                f"\n请将这些字段名添加到 src/data_loader.py 的 FIELD_MAP 字典中。"
                f"\n示例: FIELD_MAP['你的字段名'] = 'start_time'"
            )
        raise KeyError(
            f"缺少必要字段: {missing}。现有字段: {list(df.columns)}。"
            f"{hint}"
        )

    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # 保留核心字段 + 可选富特征
    keep_cols = ["start_time", "start_lng", "start_lat",
                 "end_lng", "end_lat", "end_time"]
    optional_cols = ["tripduration", "user_type", "birth_year", "gender"]
    for c in optional_cols:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols]
    n_opt = len([c for c in optional_cols if c in df.columns])
    col_names = ["time", "lng", "lat", "end_lng", "end_lat", "end_time"]
    col_names += optional_cols[:n_opt]
    assert len(col_names) == len(df.columns), (
        f"列重命名长度不匹配: col_names={len(col_names)}, df.columns={len(df.columns)}"
    )
    df.columns = col_names

    df.dropna(subset=["time", "lng", "lat", "end_lng", "end_lat"], inplace=True)
    log(f"  加载完成: {len(df):,} 条记录")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"\n共 {len(df):,} 条记录")
    print(f"时间范围: {df['time'].min()} ~ {df['time'].max()}")
