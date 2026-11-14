"""GRPO 失效诊断：对比基线模型和训练后模型的三种能力。

测三个指标：
  ① 贪心准确率      —— 最终指标（我们已经知道没涨）
  ② 采样准确率      —— 模型"能不能写出"正确答案（T=0.8, k=4）
  ③ pass@4          —— 4 次采样里至少一次对的概率
  ④ gold 的 logprob —— 正确解在模型眼里有多"顺"

判读：
  ② 涨了但①没涨 → 会写，但贪心选不中 → 需要 RFT/重排
  ② 也没涨       → 根本没学到 → 训练信号有问题
  ④ 降了         → 训练在伤模型

用法：
  python probe_diag.py \
    --baseline <Instruct路径> \
    --trained  /root/autodl-tmp/out_grpo8 \
    --n 100
"""

import argparse
import json
import os
import re

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = "/root/autodl-tmp/eval_300.jsonl"
SFT_CLEAN = "/root/autodl-tmp/sft_train_clean.jsonl"


def extract(text):
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return None if not nums else nums[-1].replace(",", "").rstrip(".")


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def gold_answer_map():
    """题目 -> 标准答案文本（用于算 gold 的 logprob）。"""
    m = {}
    with open(SFT_CLEAN, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            q = o["messages"][0]["content"].strip()
            a = o["messages"][1]["content"]
            if "####" in a:
                m[q] = a
    return m


def eval_model(path, rows, gold_map, k=4, bs=4, max_new=512):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    n_greedy_ok = 0
    n_sample_ok = 0
    n_sample_total = 0
    n_passk = 0

    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in chunk
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")

        # ① 贪心
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False
            )
        texts = tok.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for r, t in zip(chunk, texts):
            n_greedy_ok += int(same(extract(t), r["gold"]))

        # ② 采样 k 次
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=k,
            )
        texts = tok.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for j, r in enumerate(chunk):
            group = texts[j * k : (j + 1) * k]
            oks = [same(extract(t), r["gold"]) for t in group]
            n_sample_ok += sum(oks)
            n_sample_total += len(oks)
            n_passk += int(any(oks))

        print(f"  进度 {min(i+bs, len(rows))}/{len(rows)}", end="\r")

    # ④ gold 的 logprob（平均每 token）
    lp_sum, lp_n = 0.0, 0
    for i in range(0, len(rows), bs):
        chunk = rows[i : i + bs]
        for r in chunk:
            g = gold_map.get(r["question"].strip())
            if not g:
                continue
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(g, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([p_ids + a_ids], device="cuda")
            if ids.shape[1] > 1024:
                ids = ids[:, :1024]
            with torch.no_grad():
                logits = model(ids).logits[:, :-1, :].float()
                tgt = ids[:, 1:]
                picked = torch.log_softmax(logits, dim=-1).gather(
                    2, tgt.unsqueeze(-1)
                ).squeeze(-1)
                lp = picked[0, len(p_ids) - 1 :]
            lp_sum += lp.sum().item()
            lp_n += lp.numel()

    del model
    torch.cuda.empty_cache()

    return {
        "greedy": n_greedy_ok / len(rows) * 100,
        "sample": n_sample_ok / max(n_sample_total, 1) * 100,
        "passk": n_passk / len(rows) * 100,
        "gold_lp": lp_sum / max(lp_n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
    gold_map = gold_answer_map()

    print(f"探针题 {len(rows)} 道，每题采样 {args.k} 次\n")

    print("=== 基线模型 ===")
    b = eval_model(args.baseline, rows, gold_map, args.k, args.bs)
    print()
    print("=== 训练后模型 ===")
    t = eval_model(args.trained, rows, gold_map, args.k, args.bs)

    print("\n" + "=" * 66)
    print(f"{'指标':<22}{'基线':>12}{'训练后':>12}{'变化':>12}")
    print("-" * 66)
    for key, name in [
        ("greedy", "贪心准确率"),
        ("sample", "采样准确率"),
        ("passk", f"pass@{args.k}"),
        ("gold_lp", "gold 平均 logprob"),
    ]:
        bv, tv = b[key], t[key]
        if key == "gold_lp":
            print(f"{name:<22}{bv:>12.4f}{tv:>12.4f}{tv-bv:>+12.4f}")
        else:
            print(f"{name:<22}{bv:>11.2f}%{tv:>11.2f}%{tv-bv:>+11.2f}%")
    print("=" * 66)

    print("\n判读：")
    print("  采样准确率↑ 但贪心不动 → 会写，但贪心选不中")
    print("  采样也不涨             → 根本没学到")
    print("  gold logprob 变负      → 训练在伤模型")


if __name__ == "__main__":
    main()
