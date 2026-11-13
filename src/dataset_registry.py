"""
训练数据集登记与管理
====================

解决的问题
----------
原先系统把「深圳 2021-04~07」这一份数据写死在配置与界面里，换数据集后
所有描述（数据范围、小时数、模型指标、推荐度）都会失真。本模块把「训练用
哪些文件」变成可登记的清单：

  * 每个数据集 = 若干 CSV 文件的集合，带名称 / 来源 / 登记时间 / 启用开关；
  * 支持两种加入方式：
      - upload：浏览器上传（受 server.maxUploadSize 限制，适合中小数据集），
                文件落到 data/uploads/<id>/ 下；
      - local ：登记本机已有路径（目录或单文件，不拷贝，适合 GB 级数据）。
  * 只有「启用」的数据集参与训练；停用即可做对照实践，无需移动文件；
  * 全量指纹（路径+大小+修改时间）写入模型 meta，界面据此判断
    「当前模型是否还是这份数据训出来的」。

删除安全约定
------------
* source=upload：删除数据集时一并删除其上传副本（数据本就是系统内的副本）；
* source=local：只解除登记，**绝不删除用户本机文件**（可能是 12GB 级原始数据）。
"""

import os
import json
import glob
import time
import hashlib
import shutil
from datetime import datetime

import config

REGISTRY_PATH = os.path.join(config.DATA_DIR, "datasets.json")
UPLOAD_DIR = os.path.join(config.DATA_DIR, "uploads")

# 内置数据集名称：首次运行时由 data/raw 下的 CSV 自动生成
BUILTIN_NAME = "内置数据（data/raw）"

_REQUIRED_HINT = ["start_time", "start_lng", "start_lat", "end_lng", "end_lat", "end_time"]


# ══════════════════════════════ 读写 ══════════════════════════════
def _empty():
    return {"version": 1, "datasets": []}


def _load():
    if not os.path.exists(REGISTRY_PATH):
        return _empty()
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty()
    if not isinstance(data, dict) or "datasets" not in data:
        return _empty()
    return data


def _save(data):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def _new_id():
    return datetime.now().strftime("%Y%m%d%H%M%S") + str(int(time.time() * 1000) % 1000)


def _file_entry(path):
    try:
        st = os.stat(path)
        return {"path": os.path.abspath(path), "size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return {"path": os.path.abspath(path), "size": 0, "mtime": 0, "missing": True}


# ══════════════════════════════ 内置数据集 ══════════════════════════════
def ensure_default():
    """注册表不存在时，把 data/raw 下的 CSV 登记为内置数据集。

    保证升级到本版本后原有训练流程不受影响：active_files() 的结果与
    旧逻辑（扫描 RAW_DIR 全部 csv）一致。
    """
    data = _load()
    if data["datasets"]:
        return data
    files = sorted(glob.glob(os.path.join(config.RAW_DIR, "*.csv")))
    if not files:
        return data
    data["datasets"].append({
        "id": "builtin",
        "name": BUILTIN_NAME,
        "source": "local",
        "files": [_file_entry(p) for p in files],
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "enabled": True,
        "note": "升级前已在 data/raw 的数据，自动登记",
    })
    _save(data)
    return data


# ══════════════════════════════ 查询 ══════════════════════════════
def list_datasets():
    return ensure_default()["datasets"]


def _get(ds_id):
    for d in list_datasets():
        if d["id"] == ds_id:
            return d
    return None


def active_files():
    """当前参与训练的文件绝对路径列表（按数据集登记顺序）。"""
    out = []
    for d in list_datasets():
        if not d.get("enabled", True):
            continue
        for fe in d.get("files", []):
            if os.path.exists(fe["path"]):
                out.append(fe["path"])
    return out


def fingerprint(files=None):
    """文件清单指纹：路径 + 大小 + 修改时间（不含内容哈希，GB 级文件算不动）。"""
    if files is None:
        files = active_files()
    h = hashlib.sha1()
    for p in sorted(files):
        try:
            st = os.stat(p)
            h.update(f"{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{os.path.abspath(p)}|missing".encode())
    return h.hexdigest()[:16]


def summarize():
    """界面用的汇总信息。"""
    ds = list_datasets()
    act = [d for d in ds if d.get("enabled", True)]
    n_files = sum(len(d.get("files", [])) for d in act)
    total = sum(f.get("size", 0) for d in act for f in d.get("files", []))
    miss = sum(1 for d in act for f in d.get("files", []) if not os.path.exists(f["path"]))
    return {"n_datasets": len(ds), "n_active": len(act), "n_files": n_files,
            "total_bytes": total, "missing": miss}


# ══════════════════════════════ 增 ══════════════════════════════
def _validate_csv(path, sample_rows=200000):
    """校验 CSV 是否含必需字段。返回 (ok, 列名列表, 提示信息)。

    只读表头 + 少量样本，避免对 GB 级文件全量解析。
    """
    import pandas as pd
    from src.data_loader import FIELD_MAP
    try:
        head = pd.read_csv(path, nrows=5, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            head = pd.read_csv(path, nrows=5, encoding="gbk")
        except Exception as e:
            return False, [], f"无法解析（编码问题）：{e}"
    except Exception as e:
        return False, [], f"无法解析：{e}"
    cols = {c.strip(): FIELD_MAP.get(c.strip(), c.strip()) for c in head.columns}
    missing = [c for c in _REQUIRED_HINT if c not in cols.values()]
    if missing:
        return False, list(head.columns), f"缺少必需字段（映射后）：{missing}"
    return True, list(head.columns), ""


def add_uploaded(name, uploaded_files):
    """保存浏览器上传的文件并登记。返回 (数据集 dict, 校验结果列表)。"""
    ds_id = _new_id()
    target_dir = os.path.join(UPLOAD_DIR, ds_id)
    os.makedirs(target_dir, exist_ok=True)
    entries, checks = [], []
    for uf in uploaded_files:
        dest = os.path.join(target_dir, os.path.basename(uf.name))
        with open(dest, "wb") as out:
            shutil.copyfileobj(uf, out, length=1024 * 1024)
        ok, cols, msg = _validate_csv(dest)
        checks.append({"file": uf.name, "ok": ok, "msg": msg, "columns": cols})
        entries.append(_file_entry(dest))
    ds = {
        "id": ds_id,
        "name": name or f"上传数据 {ds_id}",
        "source": "upload",
        "files": entries,
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "enabled": True,
        "note": f"{len(entries)} 个上传文件",
    }
    data = _load()
    data["datasets"].append(ds)
    _save(data)
    return ds, checks


def add_local(name, path):
    """登记本机路径（目录扫描 *.csv，或单个文件），不拷贝。"""
    path = os.path.abspath(path.strip().strip('"'))
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在：{path}")
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise ValueError(f"目录下没有 CSV 文件：{path}")
    else:
        files = [path]
    checks = []
    for p in files:
        ok, cols, msg = _validate_csv(p)
        checks.append({"file": os.path.basename(p), "ok": ok, "msg": msg, "columns": cols})
    ds = {
        "id": _new_id(),
        "name": name or os.path.basename(path),
        "source": "local",
        "files": [_file_entry(p) for p in files],
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "enabled": True,
        "note": path,
    }
    data = _load()
    data["datasets"].append(ds)
    _save(data)
    return ds, checks


# ══════════════════════════════ 改 / 删 ══════════════════════════════
def set_enabled(ds_id, enabled):
    data = _load()
    for d in data["datasets"]:
        if d["id"] == ds_id:
            d["enabled"] = bool(enabled)
    _save(data)


def remove(ds_id, delete_uploaded=True):
    """移除数据集登记。

    upload 来源：同时删除 data/uploads 下的副本；
    local 来源：只解除登记，不动用户本机文件。
    """
    data = _load()
    target = None
    rest = []
    for d in data["datasets"]:
        if d["id"] == ds_id:
            target = d
        else:
            rest.append(d)
    if target is None:
        return None
    data["datasets"] = rest
    _save(data)
    if delete_uploaded and target.get("source") == "upload":
        shutil.rmtree(os.path.join(UPLOAD_DIR, target["id"]), ignore_errors=True)
    return target


def refresh_stats():
    """重新统计已登记文件的大小/修改时间（外部文件被替换后调用）。"""
    data = _load()
    for d in data["datasets"]:
        d["files"] = [_file_entry(f["path"]) for f in d.get("files", [])]
    _save(data)
    return data["datasets"]
