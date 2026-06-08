"""特征工程 + 类别编码"""
from __future__ import annotations

import time

import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

from transaction_model.constants import FRAUD_COL, FRAUD_POSITIVE_VALUES


def engineer_features(gdf):
    """执行特征工程（原地修改）

    处理内容：
    - 从 Time 列提取 Hour
    - 清洗 Amount（去 $ 和逗号）
    - 生成 _target 二值标签

    Args:
        gdf: DataFrame (cuDF 或 pandas)

    Returns:
        修改后的 gdf
    """
    print("Feature engineering...")
    t0 = time.time()
    gdf['Hour'] = gdf['Time'].str.split(':', n=1, expand=True)[0].astype(int)
    gdf['Amount'] = gdf['Amount'].str.replace('$', '', regex=False).str.replace(',', '').astype(float)
    gdf['_target'] = _make_target(gdf)
    print(f"Feature engineering: {time.time()-t0:.2f}s")
    return gdf


def _make_target(gdf):
    """从 Is Fraud? 列生成二值标签"""
    mask = gdf[FRAUD_COL] == FRAUD_POSITIVE_VALUES[0]
    for val in FRAUD_POSITIVE_VALUES[1:]:
        mask = mask | (gdf[FRAUD_COL] == val)
    return mask.astype(int)


def encode_categorical(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """使用 OrdinalEncoder 编码类别特征

    Args:
        X_train, X_val, X_test: 特征 DataFrame

    Returns:
        (X_train_enc, X_val_enc, X_test_enc, preprocessor)
    """
    print("Encoding categorical features...")
    preprocessor = make_column_transformer(
        (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         make_column_selector(dtype_include=['object', 'category'])),
        remainder='passthrough'
    )

    t0 = time.time()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)
    print(f"Encoding time: {time.time()-t0:.2f}s")

    return X_train_enc, X_val_enc, X_test_enc, preprocessor


def print_feature_summary(df, feature_cols: list[str]) -> None:
    """打印特征概要"""
    print(f"\n{len(feature_cols)}-dimensional feature set:")
    for i, col in enumerate(feature_cols):
        dtype = df[col].dtype
        nunique = df[col].nunique()
        print(f"  {i+1:2d}. {col:<20s} dtype={str(dtype):<10s} unique={nunique:,}")
