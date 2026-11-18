"""用逐题结果复算报告里的所有数字（纯 CPU，无需 GPU）。

用法：
    python verify.py

数据来源：results/dumps/*.jsonl
每个文件是逐题记录，字段：{"idx", "gold", "pred", "ok", "len"}
"""

import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(HERE, "results", "dumps")

BASE = "d1019_base.jsonl"


def load(name):
    path = os.path.join(DUMPS, name)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mcnemar_exact(b, c):
    """双侧精确 McNemar。b/c 为两个方向的不一致计数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(base, other, label):
    n = min(len(base), len(other))
    acc_b = sum(r["ok"] for r in base[:n]) / n * 100
    acc_o = sum(r["ok"] for r in other[:n]) / n * 100

    fix = sum(1 for i in range(n) if base[i]["ok"] == 0 and other[i]["ok"] == 1)
    brk = sum(1 for i in range(n) if base[i]["ok"] == 1 and other[i]["ok"] == 0)
    p = mcnemar_exact(fix, brk)

    return {
        "label": label,
        "n": n,
        "acc_before": acc_b,
        "acc_after": acc_o,
        "delta": acc_o - acc_b,
        "fix": fix,
        "break": brk,
        "net": fix - brk,
        "p": p,
    }


def fmt(r):
    sig = "YES" if r["p"] < 0.05 else "no"
    return (
        f"{r['label']:<34} {r['acc_before']:6.2f}% -> {r['acc_after']:6.2f}% "
        f"({r['delta']:+5.2f})  fix/break {r['fix']:3d}/{r['break']:3d} "
        f"net {r['net']:+4d}  p={r['p']:.4f} [{sig}]"
    )


def main():
    print("=" * 108)
    print("复算：1019 题盲测集上的逐题配对结果")
    print("注意：p 值只反映题目层面的抽样不确定性，不包含训练种子的波动。")
    print("=" * 108)

    base = load(BASE)
    n_base = sum(r["ok"] for r in base)
    print(f"\n基线（Instruct）: {n_base}/{len(base)} = {n_base/len(base)*100:.2f}%\n")

    targets = [
        ("d1019_dpo.jsonl", "DPO (Instruct)"),
        ("d1019_grpo.jsonl", "GRPO run6 (119 upd)"),
        ("d1019_grpo10.jsonl", "GRPO seedA (1426 upd)"),
        ("d1019_grpo11.jsonl", "GRPO seedC (2825 upd)"),
        ("d1019_grpo12u1426.jsonl", "GRPO seedB (1426 upd)"),
        ("d1019_grpo12.jsonl", "GRPO seedB (2872 upd)"),
        ("d1019_grpo13.jsonl", "GRPO expand (2000q x3)"),
        ("d1019_grpo13_s2.jsonl", "GRPO expand seed2 (2000q x3)"),
    ]

    rows = []
    for fn, label in targets:
        if os.path.exists(os.path.join(DUMPS, fn)):
            rows.append(compare(base, load(fn), label))

    print("vs Instruct 基线")
    print("-" * 108)
    for r in rows:
        print(fmt(r))

    # 方法间直接配对
    print("\n方法间直接配对")
    print("-" * 108)
    pairs = [
        ("d1019_dpo.jsonl", "d1019_grpo11.jsonl", "DPO  vs  GRPO seedC"),
        ("d1019_dpo.jsonl", "d1019_grpo12.jsonl", "DPO  vs  GRPO seedB(12ep)"),
        ("d1019_grpo12u1426.jsonl", "d1019_grpo12.jsonl",
         "seedB: 6ep  vs  12ep"),
        ("d1019_grpo10.jsonl", "d1019_grpo12u1426.jsonl",
         "seedA  vs  seedB (both 6ep)"),
        ("d1019_grpo12.jsonl", "d1019_grpo13.jsonl",
         "500q x12  vs  2000q x3"),
        ("d1019_grpo13.jsonl", "d1019_grpo13_s2.jsonl",
         "2000q x3: seed1 vs seed2"),
    ]
    for a, b, label in pairs:
        if os.path.exists(os.path.join(DUMPS, a)) and os.path.exists(
            os.path.join(DUMPS, b)
        ):
            ra, rb = load(a), load(b)
            n = min(len(ra), len(rb))
            fix = sum(1 for i in range(n) if ra[i]["ok"] == 0 and rb[i]["ok"] == 1)
            brk = sum(1 for i in range(n) if ra[i]["ok"] == 1 and rb[i]["ok"] == 0)
            p = mcnemar_exact(fix, brk)
            acc_a = sum(r["ok"] for r in ra[:n]) / n * 100
            acc_b = sum(r["ok"] for r in rb[:n]) / n * 100
            print(
                f"{label:<34} {acc_a:6.2f}% vs {acc_b:6.2f}% "
                f"({acc_b-acc_a:+5.2f})  fix/break {fix:3d}/{brk:3d} "
                f"net {fix-brk:+4d}  p={p:.4f}"
            )

    print("\n" + "=" * 108)
    print("说明：'fix/break' 指前一个模型错而后一个对（fix）、前一个对而后一个错（break）")
    print("=" * 108)


if __name__ == "__main__":
    main()
