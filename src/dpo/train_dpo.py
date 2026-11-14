"""阶段7：DPO 训练（全参，手写训练循环，带梯度累积）。

用法：
    python train_dpo.py --epochs 2 --lr 5e-6 --beta 0.2 --accum 8
"""

import argparse
import json
import os

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT = "/root/autodl-tmp/out_base_sft"
DATA = "/root/autodl-tmp/dpo.jsonl"


def logprob(model, p_ids, a_ids):
    """算答案部分（a_ids）在 prompt 之后的 log 概率之和。"""
    ids = torch.tensor([p_ids + a_ids], device="cuda")
    logits = model(ids).logits[:, :-1, :].float()
    targets = ids[:, 1:]
    picked = torch.log_softmax(logits, dim=-1).gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1)
    start = len(p_ids) - 1
    return picked[0, start:].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--sft", default=SFT)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--accum", type=int, default=8, help="梯度累积步数")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/root/autodl-tmp/out_dpo")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.sft)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.sft, torch_dtype=torch.bfloat16
    ).to("cuda")
    ref = AutoModelForCausalLM.from_pretrained(
        args.sft, torch_dtype=torch.bfloat16
    ).to("cuda")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(
        f"偏好对: {len(rows)} 条，beta={args.beta}, lr={args.lr}, accum={args.accum}"
    )
    print(f"每个 epoch 更新 {len(rows)//args.accum} 次参数")

    update = 0
    for epoch in range(args.epochs):
        ep_loss, ep_margin, n = 0.0, 0.0, 0
        opt.zero_grad()

        for i, r in enumerate(rows):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(r["chosen"], add_special_tokens=False)["input_ids"]
            b_ids = tok(r["rejected"], add_special_tokens=False)["input_ids"]

            lp_c = logprob(policy, p_ids, c_ids)
            lp_b = logprob(policy, p_ids, b_ids)
            with torch.no_grad():
                ref_c = logprob(ref, p_ids, c_ids)
                ref_b = logprob(ref, p_ids, b_ids)

            margin = (lp_c - ref_c) - (lp_b - ref_b)
            loss = -F.logsigmoid(args.beta * margin)

            (loss / args.accum).backward()      # 梯度累积

            ep_loss += loss.item()
            ep_margin += margin.item()
            n += 1

            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                update += 1
                if update % 20 == 0:
                    print(
                        f"epoch {epoch+1} 更新 {update:4d}  "
                        f"平均 loss={ep_loss/n:.4f}  "
                        f"平均 margin={ep_margin/n:+.3f}  "
                        f"本步 margin={margin.item():+.3f}"
                    )

        # 收尾：处理不足 accum 的残余梯度
        if len(rows) % args.accum != 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            update += 1

        print(
            f"--- epoch {epoch+1} 平均 loss = {ep_loss/n:.4f}，"
            f"平均 margin = {ep_margin/n:+.3f}，共更新 {update} 次 ---"
        )

    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("已保存到", args.out)


if __name__ == "__main__":
    main()
