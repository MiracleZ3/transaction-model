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
