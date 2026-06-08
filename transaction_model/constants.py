"""共享常量，与 YAML 配置互补（运行时不变的值）"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MCC 行业分类区间
MCC_INDUSTRY_RANGES = [
    (0, 1499, "Agricultural"), (1500, 2999, "Contracted"),
    (3000, 3299, "Airlines"), (3300, 3499, "Car Rental"),
    (3500, 3999, "Lodging"), (4000, 4799, "Transportation"),
    (4800, 4999, "Utilities"), (5000, 5599, "Retail"),
    (5600, 5699, "Clothing"), (5700, 7299, "Misc Stores"),
    (7300, 7999, "Business Services"), (8000, 8999, "Professional"),
    (9000, 9999, "Government"),
]

# UMAP 可视化参数
UMAP_VIZ_SIZE = 50_000
UMAP_AXIS_RANGE = 12
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# 欺诈标签列
FRAUD_COL = "Is Fraud?"
FRAUD_POSITIVE_VALUES = ("Yes", "1")

# ── 银联风控（risk_control_2 / new_ylformer）NDJSON 字段映射 ──────────────
# 来源：../risk_control_2/data/pre_trained/data_load.py::fields_pos
# 旧 NDJSON 每笔交易是一个 list，下标 0..19 被代码消费（20/21 在样本里存在但被跳过）。
# 加 `cups_` 前缀与旧 feature_config 的字段名保持一致。
YL_FIELDS_POS = {
    0: "cups_发卡机构地址",
    1: "cups_发卡机构银行",
    2: "cups_卡等级",
    3: "year",       # 时间 — 不入模型字段，但用于派生
    4: "month",      # 时间
    5: "day",        # 时间
    6: "hour",       # 时间
    7: "minutes",    # 时间
    8: "seconds",    # 时间
    9: "unix_timestap",  # 用于算 delta_time + 时间切分
    10: "cups_收单机构地址",
    11: "cups_收单机构银行",
    12: "cups_交易代码",
    13: "cups_交易渠道",
    14: "cups_服务点输入方式",
    15: "cups_应答码",
    16: "cups_商户类型",
    17: "cups_连接方式",
    18: "cups_受卡方名称地址",
    19: "cups_交易金额",
}

# 旧 schema 中每笔交易代码实际消费的字段数组长度（下标 0..19）。
YL_TRANS_FIELDS_LEN = 20

# 旧 NDJSON 的顶层键
YL_USER_KEY = "cert_sm3"          # 用户标识（SM3 哈希）
YL_LABEL_KEY = "label"            # 用户级二分类标签 (0/1)
YL_TRANS_KEY = "trans"            # 交易序列
YL_AMOUNT_IDX = 19                # 交易金额在 trans 数组的下标（用于金额加权）

# 卡等级已知取值（旧 schema 普通流派：白/金/钻/普；样本数据出现的是 '普'）
YL_CARD_LEVELS = ["普", "金", "白", "钻"]

# 连接方式已知取值
YL_CONN_MODES = ["直", "间"]
