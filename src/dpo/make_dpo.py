"""阶段6：用 SFT 模型采样，造 DPO 偏好数据。

用法：
    python make_dpo.py --n 200 --k 4
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

MODEL = "/root/autodl-tmp/out_base_sft"
DATA = "/root/autodl-tmp/sft_train_clean.jsonl"
OUT = "/root/autodl-tmp/dpo.jsonl"


def extract(text):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--n", type=int, default=200, help="用多少道题")
    ap.add_argument("--k", type=int, default=4, help="每题采样几个答案")
    ap.add_argument("--bs", type=int, default=8, help="一次并行几道题")
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    # 读题 + 标准答案
    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["messages"][0]["content"]
            a = obj["messages"][1]["content"]
            m = re.search(r"####\s*(-?[\d,\.]+)", a)
            if not m:
                continue
            rows.append({"q": q, "gold": m.group(1).replace(",", "")})
            if len(rows) >= args.n:
                break

    print(f"题目数: {len(rows)}，每题采样 {args.k} 个")

    samples = [[] for _ in rows]   # 每题的采样结果
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["q"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]

    for k in range(args.k):
        for i in range(0, len(rows), args.bs):
            chunk = prompts[i : i + args.bs]
            inputs = tok(chunk, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                )
            gen = out[:, inputs["input_ids"].shape[1] :]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for j, t in enumerate(texts):
                samples[i + j].append(t)
        print(f"  第 {k+1}/{args.k} 轮采样完成")

    # 配成 chosen / rejected
    pairs = []
    n_both = 0
    for r, ss in zip(rows, samples):
        oks = [s for s in ss if same(extract(s), r["gold"])]
        bads = [s for s in ss if not same(extract(s), r["gold"])]
        if oks:
            n_both += 1
        if oks and bads:
            pairs.append(
                {
                    "prompt": r["q"],
                    "chosen": oks[0],
                    "rejected": bads[0],
                }
            )

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"采样完成: {len(rows)} 题，其中 {n_both} 题至少有一个正确答案")
    print(f"可用偏好对: {len(pairs)} 条 → {args.out}")


if __name__ == "__main__":
    main()
