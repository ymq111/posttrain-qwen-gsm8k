"""多数投票评测：每题采 k 个答案，投票选出现次数最多的结果。

用途：把 pass@k 的"候选质量"兑现成准确率（不需要训练）。

用法：
  python vote_eval.py \
    --model /root/autodl-tmp/out_grpo8 \
    --dump /root/autodl-tmp/v_grpo8.jsonl \
    --n 300 --k 4
"""

import argparse
import json
import os
import re
from collections import Counter

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = "/root/autodl-tmp/eval_300.jsonl"


def extract(text):
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return None if not nums else nums[-1].replace(",", "").rstrip(".")


def norm(x):
    """规范化数字，便于投票（去掉尾部的 .0）。"""
    try:
        v = float(x)
        return str(int(v)) if v == int(v) else str(v)
    except Exception:
        return None


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]

    vote_ok = 0
    single_ok = 0
    single_total = 0
    passk_ok = 0
    dumps = []

    for i in range(0, len(rows), args.bs):
        chunk = rows[i : i + args.bs]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["question"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in chunk
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=True,
                temperature=args.temp,
                top_p=0.95,
                num_return_sequences=args.k,
            )
        texts = tok.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        for j, r in enumerate(chunk):
            group = texts[j * args.k : (j + 1) * args.k]
            preds = [extract(t) for t in group]
            nps = [norm(p) for p in preds if norm(p) is not None]

            # 单个答案的正确率
            for p in preds:
                single_ok += int(same(p, r["gold"]))
                single_total += 1

            # pass@k
            if any(same(p, r["gold"]) for p in preds):
                passk_ok += 1

            # 投票
            if nps:
                counter = Counter(nps)
                # 平票时取"出现次数最多里、第一个出现的"
                top = max(counter.values())
                winner = next(x for x in nps if counter[x] == top)
            else:
                winner = None

            ok = same(winner, r["gold"])
            vote_ok += int(ok)
            dumps.append(
                {"idx": i + j, "gold": r["gold"], "vote": winner, "ok": int(ok)}
            )

        print(f"  进度 {min(i+args.bs, len(rows))}/{len(rows)}", end="\r")

    n = len(rows)
    print("\n" + "=" * 60)
    print(f"题数 {n}，每题采样 {args.k}（T={args.temp}）")
    print(f"单次采样准确率 : {single_ok/single_total*100:.2f}%")
    print(f"pass@{args.k}        : {passk_ok/n*100:.2f}%")
    print(f"多数投票准确率 : {vote_ok/n*100:.2f}%")
    print("=" * 60)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for d in dumps:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print("逐题结果已写入", args.dump)


if __name__ == "__main__":
    main()
