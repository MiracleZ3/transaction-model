# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pre-configured tokenizer pipeline for financial transaction data.

Standard 12-field tokenizer pipeline:
    AMT  MERCH  CAT  MCC  HOUR  DOW  MONTH  CARD  CHIP  ZIP3  STATE  CUST

Optional extensions (disabled by default to keep a 12-token baseline):
    - amount_strategy="quantile"  →  data-driven amount bins via cuML
    - include_time_delta=True     →  log-compressed inter-transaction time
"""
from __future__ import annotations

try:
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    cudf = None  # type: ignore

from .pipeline import TokenizerPipeline
from .fixed_vocab import FixedVocabTokenizer
from .mapping import MappingTokenizer
from .categorical_hash import CategoricalHashTokenizer
from .numerical import NumericalTokenizerOptBin
from .timedelta import TimeDeltaTokenizer

# ── Static definitions ────────────────────────────────────────────────

KNOWN_MCCS = [
    -1, 1711, 3000, 3001, 3005, 3006, 3007, 3008, 3009, 3058, 3066,
    3075, 3132, 3144, 3174, 3256, 3260, 3359, 3387, 3389, 3390, 3393,
    3395, 3405, 3504, 3509, 3596, 3640, 3684, 3722, 3730, 3771, 3775,
    3780, 4111, 4112, 4121, 4131, 4214, 4411, 4511, 4722, 4784, 4814,
    4829, 4899, 4900, 5045, 5094, 5192, 5193, 5211, 5251, 5261, 5300,
    5310, 5311, 5411, 5499, 5533, 5541, 5621, 5651, 5655, 5661, 5712,
    5719, 5722, 5732, 5733, 5812, 5813, 5814, 5815, 5816, 5912, 5921,
    5932, 5941, 5942, 5947, 5970, 5977, 6300, 7011, 7210, 7230, 7276,
    7349, 7393, 7531, 7538, 7542, 7549, 7801, 7802, 7832, 7922, 7995,
    7996, 8011, 8021, 8041, 8043, 8049, 8062, 8099, 8111, 8931, 9402,
]

INDUSTRY_RANGES = [
    (0, 1499, "AGRICULTURAL"),
    (1500, 2999, "CONTRACTED"),
    (3000, 3299, "AIRLINES"),
    (3300, 3499, "CAR_RENTAL"),
    (3500, 3999, "LODGING"),
    (4000, 4799, "TRANSPORTATION"),
    (4800, 4999, "UTILITIES"),
    (5000, 5599, "RETAIL"),
    (5600, 5699, "CLOTHING"),
    (5700, 7299, "MISC_STORES"),
    (7300, 7999, "BUSINESS"),
    (8000, 8999, "PROFESSIONAL"),
    (9000, 9999, "GOVERNMENT"),
]

CHIP_MAPPING = {
    "SWIPE TRANSACTION": "SWIPE",
    "CHIP TRANSACTION": "CHIP",
    "ONLINE TRANSACTION": "ONLINE",
}

ALL_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP", "XX", "ONLINE",
]

AMOUNT_THRESHOLDS = [0, 10, 50, 100, 500, 1000, 5000]


class FinancialTokenizerPipeline(TokenizerPipeline):
    """Ready-to-use pipeline for the TabFormer financial-transaction dataset.

    Parameters
    ----------
    merchant_hash_size : int
        Number of hash buckets for merchant names.
    amount_strategy : {"fixed", "quantile", "uniform", "kmeans"}
        "fixed" uses 7 hard-coded dollar-amount thresholds (default).
        Others delegate to NumericalTokenizerOptBin via cuML.
    amount_bins : int
        Number of bins when using a data-driven amount_strategy.
    include_time_delta : bool
        If True, adds a TimeDeltaTokenizer step (net-new feature, off by default).
    time_delta_bins : int
        Number of log-bins for the optional time-delta step.
    """

    def __init__(
        self,
        merchant_hash_size: int = 2000,
        amount_strategy: str = "fixed",
        amount_bins: int = 10,
        include_time_delta: bool = False,
        time_delta_bins: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.merchant_hash_size = merchant_hash_size
        self.amount_strategy = amount_strategy
        self.amount_bins = amount_bins
        self.include_time_delta = include_time_delta
        self.time_delta_bins = time_delta_bins

        self._configure_steps()

    def _configure_steps(self) -> None:
        """Add the standard 12 tokenizer steps (+ optional extras)."""

        # 1. Amount
        if self.amount_strategy == "fixed":
            self.add_step("amt_val", FixedVocabTokenizer(
                prefix="AMT", min_val=0, max_val=6,
            ))
        else:
            self.add_step("amt_val", NumericalTokenizerOptBin(
                special_token="AMT",
                num_bins=self.amount_bins,
                strategy=self.amount_strategy,
            ))

        # 2. Merchant (hash)
        self.add_step("merch_hash", CategoricalHashTokenizer(
            vocab_limit=self.merchant_hash_size,
            special_token="MERCH",
        ))

        # 3. Industry category (derived from MCC via range lookup)
        self.add_step("mcc_int", MappingTokenizer(
            prefix="CAT",
            ranges=INDUSTRY_RANGES,
            default="GENERAL",
        ))

        # 4. MCC code (known codes from TabFormer + catch-all default)
        self.add_step("mcc_str", MappingTokenizer(
            prefix="MCC",
            values=[str(m) for m in KNOWN_MCCS],
            default="-1",
        ))

        # 5-7. Temporal: HOUR, DOW, MONTH
        self.add_step("hour", FixedVocabTokenizer(
            prefix="HOUR", min_val=0, max_val=23, pad_width=2,
        ))
        self.add_step("dow", FixedVocabTokenizer(
            prefix="DOW", min_val=0, max_val=6,
        ))
        self.add_step("month", FixedVocabTokenizer(
            prefix="MONTH", min_val=1, max_val=12, pad_width=2,
        ))

        # 8. Card index
        self.add_step("card", FixedVocabTokenizer(
            prefix="CARD", min_val=0, max_val=9,
        ))

        # 9. Chip type
        self.add_step("chip_upper", MappingTokenizer(
            prefix="CHIP",
            mapping=CHIP_MAPPING,
            default="UNK",
        ))

        # 10. ZIP3 region
        self.add_step("zip3", FixedVocabTokenizer(
            prefix="ZIP3", min_val=0, max_val=999, pad_width=3,
        ))

        # 11. State
        self.add_step("state_clean", MappingTokenizer(
            prefix="STATE",
            values=ALL_STATES,
            default="XX",
        ))

        # 12. Customer ID
        self.add_step("cust", FixedVocabTokenizer(
            prefix="CUST", min_val=0, max_val=2999,
        ))

        # Optional: time delta
        if self.include_time_delta:
            self.add_step("time_delta_s", TimeDeltaTokenizer(
                num_bins=self.time_delta_bins,
                special_token="TDIF",
            ))

    # ------------------------------------------------------------------
    # Preprocessing  — raw DataFrame → pipeline-ready columns
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(df):
        """Normalize columns and derive intermediate fields expected by the
        pipeline steps.  Operates in-place on a cuDF DataFrame.

        Input columns (TabFormer naming): Amount, Merchant Name, MCC,
        Year, Month, Day, Time, Card, Use Chip, Zip, Merchant State, User.

        Output adds: amt_val, merch_hash, mcc_int, mcc_str, hour, dow,
        month (numeric), chip_upper, zip3, state_clean, cust, and
        optionally time_delta_s.
        """
        if cudf is None:
            raise ImportError(
                "FinancialTokenizerPipeline.preprocess requires the 'cudf' "
                "package (GPU only). Install with: pip install cudf"
            )
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

        amt = df["amount"].astype(str).str.replace("$", "", regex=False)
        amt_f = amt.astype(float)

        df["amt_val"] = (
            (amt_f >= 10).astype("int32")
            + (amt_f >= 50).astype("int32")
            + (amt_f >= 100).astype("int32")
            + (amt_f >= 500).astype("int32")
            + (amt_f >= 1000).astype("int32")
            + (amt_f >= 5000).astype("int32")
        )

        merch_clean = (
            df["merchant_name"]
            .astype(str)
            .str.upper()
            .str.replace(r"[^A-Z0-9\s\-]", "", regex=True)
        )
        df["merch_hash"] = merch_clean.hash_values()

        mcc = df["mcc"].fillna(-1).astype(int)
        df["mcc_int"] = mcc
        df["mcc_str"] = mcc.astype(str)

        # year/month/day 入参可能是 int 或 float（parquet round-trip / cuDF 类型提升后
        # 偶尔变 float64），astype(str) 会得到 "3.0" 这类串 → strptime 失败，cuDF
        # 还会因「多种格式」直接 NotImplementedError。先强制转 int，确保拿到干净的
        # "3" / "12"。time 列允许 NaN，用 00:00 兜底。
        year_i = df["year"].fillna(0).astype("int64")
        month_i = df["month"].fillna(1).astype("int64").clip(1, 12)
        day_i = df["day"].fillna(1).astype("int64").clip(1, 31)
        time_s = df["time"].fillna("00:00").astype(str)

        date_str = (
            year_i.astype(str) + "-"
            + month_i.astype(str).str.zfill(2) + "-"
            + day_i.astype(str).str.zfill(2) + " "
            + time_s
        )
        # cuDF strptime 对异常日期（2/30、非标准时分）会直接抛 NotImplementedError；
        # 先按标准格式解析，失败再退回 pandas（errors='coerce' 把坏行变 NaT）。
        try:
            dt = cudf.to_datetime(date_str, format="%Y-%m-%d %H:%M")
        except (NotImplementedError, ValueError):
            import pandas as _pd
            col = date_str.to_pandas() if hasattr(date_str, "to_pandas") else date_str
            dt = _pd.to_datetime(col, format="%Y-%m-%d %H:%M", errors="coerce")
        # 坏日期（NaT）下 dt.hour 返回 NaN，fill 不让下游 FixedVocab 报错
        df["hour"] = dt.dt.hour.fillna(0).astype("int32")
        df["dow"] = dt.dt.dayofweek.fillna(0).astype("int32")
        df["month"] = dt.dt.month.fillna(1).astype("int32")

        df["card"] = df["card"].astype(int).clip(0, 9)
        df["chip_upper"] = df["use_chip"].astype(str).str.upper()

        zip_col = (
            df["zip"].fillna("00000").astype(str)
            .str.replace(".0", "", regex=False)
        )
        df["zip3"] = zip_col.str[:3].str.zfill(3).astype(int)

        state = (
            df["merchant_state"].fillna("XX").astype(str)
            .str.upper().str.strip()
        )
        df["state_clean"] = state.where(state != "", "XX")

        df["cust"] = df["user"].astype(int).clip(0, 2999)

        if "time_full" not in df.columns:
            df["time_full"] = dt
        df = df.sort_values(["user", "card", "time_full"])
        td = df.groupby(["user", "card"])["time_full"].diff()
        td_seconds = td.dt.total_seconds().fillna(0).clip(0)
        df["time_delta_s"] = td_seconds

        return df.reset_index(drop=True)
