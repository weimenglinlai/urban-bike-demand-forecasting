"""
全局配置 — 控制所有模块行为
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据版本号（预处理参数变更时递增，自动清空旧缓存）
DATA_VERSION = 4

# 路径配置
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# 主数据目录：按月切片 CSV（bike_orders_YYYYMMDD_YYYYMMDD_bydays.csv）
# 说明（2026-10-13 修正）：原值指向 project\项目\数据集\1，该目录不存在，
# 且数据集目录混有 CitiBike / 上海摩拜等已弃用的早期实践数据；现统一以项目内
# data/raw 作为唯一主数据源（两段按月切片 CSV 均已就位），保证缓存重建不会丢月。
RAW_DAY_DIR = RAW_DIR

for _d in [DATA_DIR, RAW_DIR, PROCESSED_DIR,
           OUTPUT_DIR, FIGURE_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

# 通用参数
GRID_SIZE = 20
# 注：原 WINDOW_SIZE = 24（早期滑动窗口方案的输入窗口）已删除。
# 序列长度现由 LSTM_PARAMS["seq_len"] / CONVLSTM_PARAMS["seq_len"] 统一承担。
# ============================================================================
# 数据切分（时间顺序，无交叉）
# ----------------------------------------------------------------------------
# 本配置采用「先按时间切出留出段(holdout)，再在剩余段内切验证集」的两级切分：
#
#     |<----------- 前 HOLDOUT_RATIO 段 ----------->|<-- 后 HOLDOUT_RATIO 段 -->|
#     |<---- CORE_RATIO ---->|<-- VAL_RATIO -->|    |          TEST            |
#               训练(核心)          验证(早停/选模)         测试(仅末次评估)
#
# 以本数据集 n=2922 小时为例：
#     TRAIN_HOLDOUT_RATIO = 0.85 → 留出段 2483 小时，测试段 439 小时
#     TRAIN_CORE_RATIO    = 0.90 → 核心 2235、验证 248（占留出段的 10%）
#     等效占比：核心 76.5% / 验证 8.5% / 测试 15.0%
#
# 注意：**不存在一个「85% 训练 + 10% 验证」的划分**。论文中应写作
#     「按时间顺序以 76.5% / 8.5% / 15.0% 切分核心训练集、验证集与测试集；
#       验证集取自核心训练段之后的 10% 留出段内部，测试集为最后 15%」，
# 以避免「85% + 10% = 95% 从何而来」的质疑。
# ============================================================================

# 留出段占比：1 - TRAIN_HOLDOUT_RATIO 即测试集占比
TRAIN_HOLDOUT_RATIO = 0.85
# 留出段内部再切给验证集的占比（早停 / 选模），不参与任何统计量估计
TRAIN_CORE_RATIO = 0.90

# 向后兼容别名（等价于上面的两级口径，勿单独修改）
TRAIN_RATIO = TRAIN_HOLDOUT_RATIO
VAL_RATIO = 1.0 - TRAIN_CORE_RATIO

RANDOM_SEED = 42
MAX_ROWS = None

# 最终特征通道数（预处理1通道 + 天气4通道，train 脚本自动检测）
FEATURE_CHANNELS = 5

# 骑行时长过滤范围（秒）
TRIP_DURATION_MIN = 60
TRIP_DURATION_MAX = 14400

# 空间池化倍数
XGB_POOL = 4          # 20 → 5×5
LSTM_POOL = 4          # 20 → 5×5
CONVLSTM_POOL = 4      # 20 → 5×5

# 原始订单流式聚合（build_hourly_demand）
ORDER_CHUNK_SIZE = 500000      # 每次读入的行数
COORD_SAMPLE_FRAC = 0.01       # 第一遍扫描估算坐标范围的采样比例（1%）

# 模型建议重训周期（天），写入 meta 供界面提示
SUGGEST_RETRAIN_DAYS = 365

# 条件基线收缩强度（经验贝叶斯逐级收缩，样本越少越向上级回退）
BASELINE_K = 6.0
# 基线「水平」的因果滑窗长度（小时）。720 = 30 天：足够覆盖节假日形状、
# 又能跟上需求水平的漂移。实际取值由验证集选择。
BASELINE_LEVEL_WINDOW = 720

# 梯度提升树参数（sklearn HistGradientBoostingRegressor）
# 注：实际参数由验证集小网格搜索选出（见 GBT_GRID），此处仅为缺省值
XGB_PARAMS = {
    "max_iter": 200,
    "max_depth": 5,
    "learning_rate": 0.08,
    "random_state": RANDOM_SEED,
}

# GBT 验证集超参搜索网格（逐区块搜索，目标为归一化残差）
GBT_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.08, 0.12],
    "max_iter": [200],
}

# 是否执行 GBT 超参网格搜索。网格规模 = |max_depth| × |learning_rate| × |max_iter|
#   × pg²（当前 3×3×1×25 = 225 次逐区块拟合），是整轮训练中占比最大的单项开销。
# 置 False 时跳过搜索、直接采用 XGB_PARAMS 中的缺省值，用于：
#   1. 调试与冒烟测试（只需一个能用的模型，不需要最优参数）；
#   2. 数据或特征变更后的快速迭代，待结构稳定后再开搜索定稿。
# 正式结果必须置 True 重训，否则 meta["gbt_best_params"] 为空、与论文口径不符。
GBT_SEARCH = True

# 多层感知机参数（消融对照：无序列机制的神经网络）
# 说明：MLP 已完成「隔离序列机制贡献」的历史使命，论文主线与前端由 Transformer 接替；
# 此处保留其配置、类与权重，作为内部消融实践的活证据，供评委追溯（见 README 附录）。
MLP_PARAMS = {
    "hidden": [128, 64],
    "dropout": 0.10,
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 300,
    "early_stop_patience": 30,
    "lr_reduce_patience": 8,
    "lr_reduce_factor": 0.5,
    "weight_decay": 1e-5,
    "val_mode": "both",
}

# Transformer 参数（残差序列 Transformer Encoder，双通道）
# 与 LSTM 同轨：同样吃 (seq_len+1) 步残差序列，同样带外生直连分支，
# 只是把循环单元换成自注意力 + 位置编码，用于建模更长程的依赖。
TRANSFORMER_PARAMS = {
    "seq_len": 24,             # 自注意力覆盖最近 24 小时残差
    "d_model": 64,             # 注意力内部维度（须能被 nhead 整除）
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 128,
    "dropout": 0.1,
    "ctx_hidden": [128, 64],   # 外生直连分支（z(t) → 残差）的隐层，保证冷启动不劣于纯外生映射
    "history_dropout": 0.3,    # 训练期随机丢弃历史残差，使冷启动（纯未来预测）可用
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 300,
    "early_stop_patience": 30,
    "lr_reduce_patience": 8,
    "lr_reduce_factor": 0.5,
    "weight_decay": 1e-5,
    "val_mode": "both",
}

# LSTM 参数（残差序列模型）
LSTM_PARAMS = {
    "seq_len": 24,             # 循环 24 小时残差
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "ctx_hidden": [128, 64],   # 外生直连分支（z(t) → 残差）的隐层，保证冷启动不劣于纯外生映射
    "history_dropout": 0.3,    # 训练期随机丢弃历史残差，使冷启动（纯未来预测）可用
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 300,
    "early_stop_patience": 30,
    "lr_reduce_patience": 8,
    "lr_reduce_factor": 0.5,
    "weight_decay": 1e-5,
    "val_mode": "both",
}

# ConvLSTM 参数（残差张量 + FiLM 条件调制）
CONVLSTM_PARAMS = {
    "seq_len": 24,
    "hidden_channels": [32, 32],
    "kernel_size": (3, 3),
    "ctx_hidden": [128, 64],
    "dropout": 0.2,
    "history_dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 300,
    "early_stop_patience": 30,
    "lr_reduce_patience": 8,
    "lr_reduce_factor": 0.5,
    "weight_decay": 1e-5,
    "val_mode": "both",
    "max_training_minutes": 180,
}
