"""逐题配对比较 + McNemar 检验。

回答的问题：
  A = 基线错、新模型对 的题数（改对了）
  B = 基线对、新模型错 的题数（改错了）
  净变化 = (A - B) / 总数
  McNemar 精确检验：这个净变化是否显著

用法：
  python compare_dumps.py dump_sft.jsonl dump_dpo.jsonl
"""

import json
import math
import sys


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def mcnemar_exact(a, b):
    """双侧精确 McNemar 检验（二项分布）。"""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    if len(sys.argv) < 3:
        print("用法: python compare_dumps.py 基线.jsonl 对比.jsonl")
        return

    base = load(sys.argv[1])
    other = load(sys.argv[2])

    if len(base) != len(other):
        print(f"长度不一致: {len(base)} vs {len(other)}")
        return

    n = len(base)
    both_ok = both_bad = a = b = 0
    for x, y in zip(base, other):
        if x["ok"] and y["ok"]:
            both_ok += 1
        elif not x["ok"] and not y["ok"]:
            both_bad += 1
        elif not x["ok"] and y["ok"]:
            a += 1        # 改对
        else:
            b += 1        # 改错

    acc_base = sum(x["ok"] for x in base) / n
    acc_other = sum(x["ok"] for x in other) / n
    p = mcnemar_exact(a, b)

    print(f"题数: {n}")
    print(f"基线准确率: {acc_base*100:.2f}%")
    print(f"对比准确率: {acc_other*100:.2f}%")
    print(f"净变化    : {(acc_other-acc_base)*100:+.2f} 个百分点")
    print()
    print(f"两者都对: {both_ok}")
    print(f"两者都错: {both_bad}")
    print(f"A 改对  : {a}")
    print(f"B 改错  : {b}")
    print()
    print(f"McNemar 精确检验 p = {p:.4f}", "→ 显著 ✅" if p < 0.05 else "→ 不显著（差异可能来自波动）")

    lb = sum(x["len"] for x in base) / n
    lo = sum(x["len"] for x in other) / n
    print()
    print(f"平均输出长度: {lb:.0f} → {lo:.0f} 字符 ({(lo-lb)/lb*100:+.1f}%)")


if __name__ == "__main__":
    main()
