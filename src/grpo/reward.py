"""D9：GRPO 的奖励函数。

规则（可加和）：
    答案正确        +1.0
    有 #### 格式     +0.2
    答案错误        −0.5

所以：
    对 + 有格式  = +1.2
    对 + 无格式  = +1.0
    错 + 有格式  = −0.3
    错 + 无格式  = −0.5

跑法：
    python reward.py          # 用 dpo_full.jsonl 里的 chosen/rejected 验证
"""

import json
import re

# ---------- 三个基础函数 ----------


def extract(text):
    """抽答案：优先 #### 后的数字，没有就退回最后一个数字。"""
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same(pred, gold):
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        return False


def reward(text, gold):
    """给一个答案打分。返回 (分数, 明细)。"""
    has_hash = "####" in text
    pred = extract(text)
    correct = same(pred, gold)

    score = 0.0
    if correct:
        score += 1.0
    else:
        score -= 0.5
    if has_hash:
        score += 0.2

    detail = {
        "correct": correct,
        "has_hash": has_hash,
        "pred": pred,
        "gold": gold,
        "score": round(score, 2),
    }
    return score, detail


# ---------- 自测 ----------


def main():
    # ① 手工用例：覆盖各种情况
    cases = [
        ("Natalia sold 48/2 = 24 ... #### 72", "72", 1.2, "对 + 有格式"),
        ("Natalia sold 48/2 = 24 ... 72", "72", 1.0, "对 + 无格式"),
        ("Natalia sold 48*4 = 192 ... #### 192", "72", -0.3, "错 + 有格式"),
        ("I don't know", "72", -0.5, "错 + 无格式"),
        ("#### -5", "-5", 1.2, "负数答案"),
        ("#### 3.5", "3.5", 1.2, "小数答案"),
        ("#### 1,234", "1234", 1.2, "带逗号的大数"),
    ]

    print("=" * 62)
    print("手工用例")
    print("=" * 62)
    ok_all = True
    for text, gold, expect, note in cases:
        s, d = reward(text, gold)
        flag = "✅" if abs(s - expect) < 1e-9 else "❌"
        if flag == "❌":
            ok_all = False
        print(f"{flag} {note:16s} 期望 {expect:+.1f}  实际 {s:+.1f}   pred={d['pred']}")

    print("\n手工用例全部通过" if ok_all else "\n有失败用例，检查 reward()")

    # ② 用真实数据验证：chosen 应该得高分，rejected 得低分
    path = "/root/autodl-tmp/dpo_full.jsonl"
    try:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    except FileNotFoundError:
        print(f"\n（跳过真实数据验证：找不到 {path}）")
        return

    c_scores, r_scores = [], []
    for r in rows:
        gold_m = re.search(r"####\s*(-?[\d,\.]+)", r["prompt"])
        # 从原始训练数据取 gold 更稳，这里用 chosen 里的 #### 反推
        m = re.search(r"####\s*(-?[\d,\.]+)", r["chosen"])
        if not m:
            continue
        gold = m.group(1).replace(",", "")
        c_scores.append(reward(r["chosen"], gold)[0])
        r_scores.append(reward(r["rejected"], gold)[0])

    n = len(c_scores)
    print("\n" + "=" * 62)
    print(f"真实数据验证（{n} 对）")
    print("=" * 62)
    print(f"chosen   平均奖励 = {sum(c_scores)/n:+.3f}")
    print(f"rejected 平均奖励 = {sum(r_scores)/n:+.3f}")
    gap = sum(c_scores) / n - sum(r_scores) / n
    print(f"两者差距            = {gap:+.3f}")

    n_disc = sum(1 for c, r in zip(c_scores, r_scores) if c > r)
    print(f"chosen 得分更高的比例 = {n_disc/n*100:.1f}%")

    print("\n判读：")
    print("  chosen 应接近 +1.2，rejected 应接近 −0.3")
    print("  差距应明显为正，chosen 得分更高的比例应接近 100%")


if __name__ == "__main__":
    main()
