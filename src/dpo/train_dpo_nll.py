"""阶段7：DPO + chosen NLL 混合损失（A/B/C 三组对照）。

三组用同一份数据、同一个起点、同样的步数，只改损失：

  A 组（只学 chosen，RFT 对照）:
      python train_dpo_nll.py --dpo_weight 0 --nll_weight 1 --out out_rft

  B 组（标准 DPO）:
      python train_dpo_nll.py --dpo_weight 1 --nll_weight 0 --out out_dpo_b

  C 组（DPO + NLL）:
      python train_dpo_nll.py --dpo_weight 1 --nll_weight 1 --out out_dpo_c

说明：
  NLL = −(chosen 的 logprob 之和) ÷ (chosen 的有效 token 数)
  按回答长度归一化（与 DPO 的求和口径不同，这是有意的）
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
DATA = "/root/autodl-tmp/dpo_full.jsonl"


def logprob_sum(model, p_ids, a_ids):
    """答案部分的 log 概率之和。"""
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
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--dpo_weight", type=float, default=1.0)
    ap.add_argument("--nll_weight", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/root/autodl-tmp/out_dpo_c")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.sft)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.sft, torch_dtype=torch.bfloat16
    ).to("cuda")

    # 只有用到 DPO 项时才需要 reference 模型（省显存）
    ref = None
    if args.dpo_weight > 0:
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
        f"偏好对: {len(rows)} 条 | dpo_weight={args.dpo_weight} "
        f"nll_weight={args.nll_weight} | lr={args.lr} beta={args.beta} accum={args.accum}"
    )

    update = 0
    for epoch in range(args.epochs):
        ep_dpo, ep_nll, ep_margin, n = 0.0, 0.0, 0.0, 0
        opt.zero_grad()

        for i, r in enumerate(rows):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(r["chosen"], add_special_tokens=False)["input_ids"]

            loss = torch.zeros((), device="cuda")
            margin_val = 0.0

            # DPO 项
            if args.dpo_weight > 0:
                b_ids = tok(r["rejected"], add_special_tokens=False)["input_ids"]
                lp_c = logprob_sum(policy, p_ids, c_ids)
                lp_b = logprob_sum(policy, p_ids, b_ids)
                with torch.no_grad():
                    ref_c = logprob_sum(ref, p_ids, c_ids)
                    ref_b = logprob_sum(ref, p_ids, b_ids)
                margin = (lp_c - ref_c) - (lp_b - ref_b)
                dpo_loss = -F.logsigmoid(args.beta * margin)
                loss = loss + args.dpo_weight * dpo_loss
                ep_dpo += dpo_loss.item()
                margin_val = margin.item()
            else:
                lp_c = None

            # NLL 项（按 token 数归一化）
            if args.nll_weight > 0:
                if lp_c is None:
                    lp_c = logprob_sum(policy, p_ids, c_ids)
                nll = -lp_c / max(len(c_ids), 1)
                loss = loss + args.nll_weight * nll
                ep_nll += nll.item()

            (loss / args.accum).backward()

            ep_margin += margin_val
            n += 1

            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                update += 1
                if update % 20 == 0:
                    print(
                        f"epoch {epoch+1} 更新 {update:4d}/{len(rows)//args.accum}  "
                        f"dpo_loss={ep_dpo/max(n,1):.4f}  "
                        f"nll={ep_nll/max(n,1):.4f}  "
                        f"margin={ep_margin/max(n,1):+.3f}"
                    )

        if len(rows) % args.accum != 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            update += 1

        print(
            f"--- epoch {epoch+1}  dpo_loss={ep_dpo/max(n,1):.4f}  "
            f"nll={ep_nll/max(n,1):.4f}  margin={ep_margin/max(n,1):+.3f}  "
            f"更新 {update} 次 ---"
        )

    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("已保存到", args.out)


if __name__ == "__main__":
    main()
