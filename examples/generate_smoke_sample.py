"""生成 examples/sample_data/smoke.jsonl —— 仓库内置冒烟测试样例。

特点：
  - 20 个用户（cert_sm3 = u1..u20）
  - 10 个正样本 (label=1)、10 个负样本 (label=0)
  - 每用户 ~30 笔交易，时间戳单调递增
  - 字段值从一组扩张词表里随机抽，保证去重（每条交易不同）
  - 严格满足 risk_control_2 / YL_FIELDS_POS 的 22 字段格式
    （前 20 字段被代码消费，21-22 字段是 16 进制串占位符）

用途：clone 仓库后跑 pytest / step_01/02/06 时不必依赖外部数据。

跑法：
    python examples/generate_smoke_sample.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path


# 字段值词汇表（覆盖 YL_FIELDS_POS 各字段的合理取值）
FAKA_ADDR    = ["0000", "0010", "0020", "0030", "1111", "2222"]
FAKA_BANK    = ["工商银行", "建设银行", "招商银行", "中国银行", "宁夏银行",
                "农业银行", "交通银行", "中信银行"]
CARD_LEVEL   = ["普", "金", "白", "钻"]
SHOUDAN_ADDR = ["银联代发", "广东省", "北京", "上海", "深圳", "杭州", "成都"]
SHOUDAN_BANK = ["支付宝", "财付通", "微信支付", "美团", "京东", "滴滴"]
TRX_CODE     = ["消费类", "贷记", "转账", "查询", "取现", "退款"]
CHANNEL      = ["无线", "网上", "ATM", "柜台", "POS"]
POS_INPUT    = ["手工,不含PIN", "免密", "磁条", "芯片", "NFC"]
RESP_CODE    = ["承兑或交易成功, 成功", "资金不足, 失败",
                "持卡人超限", "终端故障", "校验错误"]
MCC          = ["6066", "6071", "5812", "5411", "7011",
                "未列入其他代码的商业服务", "非金融机构－外币兑换、非电子转账的汇票"]
CONN_MODE    = ["直", "间"]
MERCHANT     = [
    "支付宝-转账/**华", "支付宝-消费/嘉兴繁顶售卖机",
    "财付通支付科技有限公司", "微信支付/微信转账",
    "美团外卖/中关村店", "京东-家电", "滴滴-打车", "中石化加油站",
]


def gen_txn(rng, ts_unix, is_positive_run):
    """生成一笔交易。ts_unix 是该笔交易的 unix 秒。

    is_positive_run → True: 倾向于取大金额 + 失败响应码（模拟风险交易）；
    False: 随机取（常规交易）。仅用于让 label 与特征略有相关，方便测试区分。
    """
    import datetime as _dt
    dt = _dt.datetime.fromtimestamp(ts_unix)

    amount_pool = ([5000, 10000, 50000, 100000, 300000] if is_positive_run
                   else [34, 1750, 100, 5000, 50, 1000, 200])

    txn = [
        rng.choice(FAKA_ADDR),
        rng.choice(FAKA_BANK),
        rng.choice(CARD_LEVEL),
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second,
        float(ts_unix),
        rng.choice(SHOUDAN_ADDR),
        rng.choice(SHOUDAN_BANK),
        rng.choice(TRX_CODE),
        rng.choice(CHANNEL),
        rng.choice(POS_INPUT),
        rng.choice(RESP_CODE) if not is_positive_run else "资金不足, 失败",
        rng.choice(MCC),
        rng.choice(CONN_MODE),
        rng.choice(MERCHANT),
        rng.choice(amount_pool),
        # 字段 20/21：与 risk_control_2 sample 一致的 16 进制串占位符
        f"{rng.randrange(0x10000):05X}",
        f"{rng.randrange(0x100000000):08X}",
    ]
    return txn


def gen_user(rng, user_id, label, base_start_ts):
    """生成一个用户的完整历史。"""
    n_txn = rng.randint(20, 35)
    # 时间戳按 ~1-7 天间隔递增
    cur = base_start_ts
    txns = []
    for _ in range(n_txn):
        gap = rng.randint(3600 * 6, 3600 * 24 * 7)   # 6h ~ 7d
        cur += gap
        txns.append(gen_txn(rng, cur, is_positive_run=bool(label)))

    return {
        "cert_sm3": user_id,
        "cert_type": "cert",
        "trans": txns,
        "label": label,
    }


def main():
    rng = random.Random(42)
    base_ts = 1712067737.0   # 2024-04-02 22:22:17 UTC，与原样本起点对齐

    records = []
    for i in range(1, 21):
        # u1..u10 → label=1；u11..u20 → label=0
        label = 1 if i <= 10 else 0
        records.append(
            gen_user(rng, f"u{i}", label, base_ts + i * 86400)
        )

    out = Path(__file__).parent / "sample_data" / "smoke.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} users → {out}")
    print(f"  Size: {out.stat().st_size / 1024:.1f} KB")
    print(f"  Labels: 1={sum(r['label']==1 for r in records)}, "
          f"0={sum(r['label']==0 for r in records)}")
    print(f"  Total transactions: {sum(len(r['trans']) for r in records)}")
    print(f"  Distinct sequences: "
          f"{len(set(json.dumps(r['trans'], ensure_ascii=False, sort_keys=True) for r in records))}")


if __name__ == "__main__":
    main()
