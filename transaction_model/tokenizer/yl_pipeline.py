"""银联（risk_control_2 / ylformer）交易流水 tokenizer pipeline。

复用现有 6 个 tokenizer 原语（fixed_vocab / mapping / categorical_hash /
numerical / timedelta / pipeline 编排），把 15 个银联字段挂成 ~15-17 个 step，
产出与 ``FinancialTokenizerPipeline`` 同结构的 token 字符串序列。

设计要点（详见 ``upgrade/ylformer.md`` 附录 A）：

  - 卡等级、连接方式：闭集（MappingTokenizer, 静态 values）
  - 交易代码/渠道/服务点输入方式/应答码/商户类型：低基数类别，fit 自数据
        → MappingTokenizer(values=None, default="UNK") 在 build_vocab 时从数据学习
  - 发卡/收单机构地址/银行：高基数 → CategoricalHashTokenizer
  - 受卡方名称地址：自由文本（旧走 Trie 子词）→ 这里简化为 hash 桶
        （路线 A 不引入 Trie，路线 B 才保留）
  - 交易金额：NumericalTokenizerOptBin(strategy="quantile") 数据驱动分桶，
        替代旧 feature_config['cups_交易金额']['bins'] 的固定桶
  - hour / dow / period：固定区间 FixedVocabTokenizer
  - 可选 delta_time：TimeDeltaTokenizer 补偿无 GPT2 频率编码（include_time_delta=True）

输入 DataFrame 期望含如下列（来自 ``ndjson_loader.expand_records_to_dataframe``）::

    cups_发卡机构地址, cups_发卡机构银行, cups_卡等级,
    cups_收单机构地址, cups_收单机构银行,
    cups_交易代码, cups_交易渠道, cups_服务点输入方式, cups_应答码,
    cups_商户类型, cups_连接方式, cups_受卡方名称地址, cups_交易金额,
    cups_星期, cups_时间段,
    year, month, day, hour, minutes, seconds,
    unix_timestap, cert_sm3, [label], seq_order

调用方式与 ``FinancialTokenizerPipeline`` 完全一致::

    pip = YLPipeline(merchant_hash_size=4000, amount_strategy="quantile")
    df = pip.preprocess(raw_df)   # 需 cuDF
    pip.fit(df)
    token_df = pip.transform(df)
    corpus_lines = pip.to_corpus_lines(token_df, df, group_cols=["cert_sm3"])
"""
from __future__ import annotations

from typing import Optional

try:
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    cudf = None  # type: ignore

from .categorical_hash import CategoricalHashTokenizer
from .fixed_vocab import FixedVocabTokenizer
from .mapping import MappingTokenizer
from .numerical import NumericalTokenizerOptBin
from .pipeline import TokenizerPipeline
from .timedelta import TimeDeltaTokenizer

from transaction_model.constants import YL_CARD_LEVELS, YL_CONN_MODES


# ── 已知闭集（从 risk_control_2 schema + sample 数据推断）──────────────

# 卡等级：普/金/白/钻（4 类）
_CARD_LEVEL_VALUES = list(YL_CARD_LEVELS)

# 连接方式：直/间（2 类）
_CONN_MODE_VALUES = list(YL_CONN_MODES)


class YLPipeline(TokenizerPipeline):
    """银联风控流水线。

    Parameters
    ----------
    merchant_hash_size : int
        高基数机构地址/银行字段的 hash 桶数。
    merch_name_hash_size : int
        「受卡方名称地址」（文本字段）的 hash 桶数；默认与 merchant_hash_size 相同。
        旧 schema 该字段是 text 走 Trie 子词（~20000），这里用 hash 近似。
    amount_strategy : {"fixed", "quantile", "uniform", "kmeans"}
        金额分桶策略。默认 "quantile"（数据驱动，对漂移鲁棒）。
    amount_bins : int
        金额桶数（对齐旧 feature_config num_class=20）。
    amount_thresholds : list[float], optional
        当 amount_strategy="fixed" 时的固定阈值。None 时用默认 7 档。
    include_time_delta : bool
        是否加入相邻交易时间差 step（True 推荐用于无频率编码的 decoder-only 模型）。
    time_delta_bins : int
        时间差 log-bucket 数。
    """

    def __init__(
        self,
        merchant_hash_size: int = 4000,
        merch_name_hash_size: Optional[int] = None,
        amount_strategy: str = "quantile",
        amount_bins: int = 20,
        amount_thresholds: Optional[list] = None,
        include_time_delta: bool = True,
        time_delta_bins: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if amount_strategy not in {"fixed", "quantile", "uniform", "kmeans"}:
            raise ValueError(f"Unknown amount_strategy: {amount_strategy}")

        self.merchant_hash_size = merchant_hash_size
        self.merch_name_hash_size = merch_name_hash_size or merchant_hash_size
        self.amount_strategy = amount_strategy
        self.amount_bins = amount_bins
        self.amount_thresholds = amount_thresholds
        self.include_time_delta = include_time_delta
        self.time_delta_bins = time_delta_bins

        self._configure_steps()

    # ------------------------------------------------------------------
    # Step 装配 — 对应 FinancialTokenizerPipeline._configure_steps
    # ------------------------------------------------------------------

    def _configure_steps(self) -> None:
        """添加银联 schema 的 ~17 个 tokenizer step。

        字段映射详见 upgrade/ylformer.md 附录 A。
        """
        # 1. 交易金额（numerical）
        if self.amount_strategy == "fixed":
            # 静态阈值分桶（与 FinancialTokenizerPipeline 风格一致）
            self.add_step(
                "cups_交易金额",
                FixedVocabTokenizer(
                    prefix="AMT",
                    min_val=0,
                    max_val=max(1, len(self.amount_thresholds or []) - 1),
                )
                if self.amount_thresholds
                else FixedVocabTokenizer(
                    prefix="AMT", min_val=0, max_val=6
                ),
            )
        else:
            self.add_step(
                "cups_交易金额",
                NumericalTokenizerOptBin(
                    special_token="AMT",
                    num_bins=self.amount_bins,
                    strategy=self.amount_strategy,
                ),
            )

        # 2. 受卡方名称地址（高基数文本 → hash）
        self.add_step(
            "cups_受卡方名称地址",
            CategoricalHashTokenizer(
                vocab_limit=self.merch_name_hash_size,
                special_token="MERCH_NAME",
            ),
        )

        # 3. 商户类型（低基数）
        self.add_step(
            "cups_商户类型",
            MappingTokenizer(prefix="MCC", default="UNK"),
        )

        # 4. 卡等级（闭集 4 类）
        self.add_step(
            "cups_卡等级",
            MappingTokenizer(
                prefix="CARDLVL",
                values=_CARD_LEVEL_VALUES,
                default="UNK",
            ),
        )

        # 5. 交易代码（低基数）
        self.add_step(
            "cups_交易代码",
            MappingTokenizer(prefix="TRXCODE", default="UNK"),
        )

        # 6. 交易渠道（低基数）
        self.add_step(
            "cups_交易渠道",
            MappingTokenizer(prefix="CHANNEL", default="UNK"),
        )

        # 7. 服务点输入方式（低基数）
        self.add_step(
            "cups_服务点输入方式",
            MappingTokenizer(prefix="POSINPUT", default="UNK"),
        )

        # 8. 应答码（低基数）
        self.add_step(
            "cups_应答码",
            MappingTokenizer(prefix="RESPCODE", default="UNK"),
        )

        # 9. 连接方式（闭集 直/间）
        self.add_step(
            "cups_连接方式",
            MappingTokenizer(
                prefix="CONNMODE",
                values=_CONN_MODE_VALUES,
                default="UNK",
            ),
        )

        # 10-11. 发卡 / 收单机构地址（高基数 → hash）
        self.add_step(
            "cups_发卡机构地址",
            CategoricalHashTokenizer(
                vocab_limit=self.merchant_hash_size,
                special_token="FAKA_ADDR",
            ),
        )
        self.add_step(
            "cups_收单机构地址",
            CategoricalHashTokenizer(
                vocab_limit=self.merchant_hash_size,
                special_token="SHOUDAN_ADDR",
            ),
        )

        # 12-13. 发卡 / 收单银行（高基数 → hash）
        self.add_step(
            "cups_发卡机构银行",
            CategoricalHashTokenizer(
                vocab_limit=self.merchant_hash_size,
                special_token="FAKA_BANK",
            ),
        )
        self.add_step(
            "cups_收单机构银行",
            CategoricalHashTokenizer(
                vocab_limit=self.merchant_hash_size,
                special_token="SHOUDAN_BANK",
            ),
        )

        # 14. 时间段（派生，闭集 0..3）
        self.add_step(
            "cups_时间段",
            FixedVocabTokenizer(prefix="PERIOD", min_val=0, max_val=3),
        )

        # 15. 星期（派生，闭集 0..6）
        self.add_step(
            "cups_星期",
            FixedVocabTokenizer(prefix="DOW", min_val=0, max_val=6),
        )

        # 16-17. 时间 delta（可选 — 推荐保留以补偿无 GPT2 频率编码）
        if self.include_time_delta:
            self.add_step(
                "time_delta_s",
                TimeDeltaTokenizer(
                    num_bins=self.time_delta_bins,
                    special_token="TDIF",
                ),
            )

    # ------------------------------------------------------------------
    # Preprocess — 原 DataFrame → pipeline-ready 列
    # ------------------------------------------------------------------

    def preprocess(self, df):
        """清洗 + 派生 ``time_delta_s`` 等运行时列。

        输入期望已经是 ``ndjson_loader.expand_records_to_dataframe`` 的输出
        （已含 cups_* 列、cert_sm3、unix_timestap 等）。本方法做的工作是

          - 把金额列（cups_交易金额）规整成 float
          - 把文本字段（cups_受卡方名称地址）转大写字符串
          - 把机构地址/银行字段转字符串
          - 计算相邻交易时间差 ``time_delta_s``（秒，按 cert_sm3 分组）
          - 给每个 hash step 预计算一个整数 hash 列
            （CategoricalHashTokenizer.tokenize 期望传入已 hash 的整数列）

        与 ``FinancialTokenizerPipeline.preprocess`` 一样，需要 cuDF。
        """
        if cudf is None:
            raise ImportError(
                "YLPipeline.preprocess requires the 'cudf' "
                "package (GPU only). Install with: pip install cudf"
            )

        df = df.copy()

        # 1. 金额 → float（旧 schema 已是 int/float，但统一防漂移）
        if "cups_交易金额" in df.columns:
            df["cups_交易金额"] = (
                df["cups_交易金额"].astype("str").str.replace(",", "", regex=False)
                .astype("float64")
            )

        # 2. 受卡方名称 → 大写（hash 用）
        if "cups_受卡方名称地址" in df.columns:
            df["cups_受卡方名称地址"] = (
                df["cups_受卡方名称地址"].astype("str").str.upper()
                .str.replace(r"[^A-Z0-9\u4e00-\u9fff\s\-/]", "", regex=True)
            )

        # 3. 高基数机构字段转字符串（hash 需要 str）
        for col in [
            "cups_发卡机构地址", "cups_发卡机构银行",
            "cups_收单机构地址", "cups_收单机构银行",
        ]:
            if col in df.columns:
                df[col] = df[col].astype("str").str.upper().str.strip()

        # 4. 低基数类别字段转字符串（fit 学习唯一值需要 str）
        for col in [
            "cups_商户类型", "cups_交易代码", "cups_交易渠道",
            "cups_服务点输入方式", "cups_应答码",
            "cups_卡等级", "cups_连接方式",
        ]:
            if col in df.columns:
                df[col] = df[col].astype("str")

        # 5. 派生时间差（按 cert_sm3 内的 seq_order 排序）
        #    与 FinancialTokenizerPipeline.preprocess 的 time_delta 一致：
        #    首笔交易 delta=0，其余 = (ts[i]-ts[i-1]) 秒。
        if "unix_timestap" in df.columns and "cert_sm3" in df.columns:
            sort_cols = ["cert_sm3"]
            if "seq_order" in df.columns:
                sort_cols.append("seq_order")
            df = df.sort_values(sort_cols).reset_index(drop=True)

            ts = df["unix_timestap"].astype("float64")
            diff = df.groupby("cert_sm3")["unix_timestap"].astype("float64").diff()
            # cudf / pandas 都有 .diff()；不存在则用 fillna(0)。秒数。
            td_seconds = (diff.fillna(0).clip(lower=0)).astype("float64")
            df["time_delta_s"] = td_seconds

        # 6. 为每个 hash step 预计算整数 hash 列
        #    CategoricalHashTokenizer.tokenize 期望输入是「已 hash 的整数」
        #    （与 FinancialTokenizerPipeline.preprocess 一致：merch_hash 用 hash_values()）
        for step_name in self.tokenizer_order:
            tok = self.steps.get(step_name)
            if isinstance(tok, CategoricalHashTokenizer):
                if step_name in df.columns:
                    df[step_name] = df[step_name].hash_values()

        return df
