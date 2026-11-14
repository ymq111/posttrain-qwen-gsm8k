"""准备扩题实验的数据：取 GSM8K train 前 2000 题。

原 500 题（grpo10/11/12 用的）保留在开头，新增第 501-2000 题。
注意：train_grpo2.py 的 --n 参数取前 n 条，所以直接用 --n 2000 即可，
不需要单独生成文件。本脚本只用于核查数据量与去重情况。

用法：
  python prep_expand.py
"""

import json
import re
from collections import Counter

SRC = "/root/autodl-tmp/sft_train_clean.jsonl"
EVAL300 = "/root/autodl-tmp/eval_300.jsonl"
EVAL1019 = "/root/autodl-tmp/eval_1019.jsonl"


def norm(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def main():
    rows = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            q = o["messages"][0]["content"]
            a = o["messages"][1]["content"]
            m = re.search(r"####\s*(-?[\d,\.]+)", a)
            rows.append({"q": q, "gold": m.group(1).replace(",", "") if m else None})

    print(f"训练池总数: {len(rows)}")
    print(f"缺答案的条数: {sum(1 for r in rows if r['gold'] is None)}")

    # 取前 2000
    use = rows[:2000]
    print(f"\n扩题实验将使用: {len(use)} 题（原 500 + 新 {len(use)-500}）")

    # 与评测集去重检查
    ev = set()
    for p in (EVAL300, EVAL1019):
        with open(p, encoding="utf-8") as f:
            for line in f:
                ev.add(norm(json.loads(line)["question"]))

    dup = sum(1 for r in use if norm(r["q"]) in ev)
    print(f"与评测集重叠: {dup} 条 {'✅ 无重叠' if dup == 0 else '❌ 需处理'}")

    # 内部重复检查
    c = Counter(norm(r["q"]) for r in use)
    dups = {k: v for k, v in c.items() if v > 1}
    print(f"题目内部重复: {len(dups)} 组 {'✅ 无重复' if not dups else ''}")

    # 答案分布抽样
    print("\n前 5 条预览：")
    for r in use[:5]:
        print(f"  gold={r['gold']:<8} q={r['q'][:70]}...")

    print("\n最后 3 条（第 1998-2000）预览：")
    for r in use[-3:]:
        print(f"  gold={r['gold']:<8} q={r['q'][:70]}...")

    # 统计答案长度
    lens = [len(r["gold"]) for r in use if r["gold"]]
    print(f"\n答案长度：最短 {min(lens)}，最长 {max(lens)}，中位 {sorted(lens)[len(lens)//2]}")

    print("\n" + "=" * 60)
    print("结论：直接使用 --n 2000 即可，无需单独生成数据文件。")
    print("=" * 60)


if __name__ == "__main__":
    main()
