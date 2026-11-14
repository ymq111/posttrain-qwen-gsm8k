"""GRPO v2：修复三个问题。

修复点（对照《解答3》）：
  ① 按第一个 EOS 截断，避免 PAD（= EOS）混入 logprob
  ② KL 用 k3 估计量（expm1(d) - d），提供真正的拉回力；默认关闭（kl_weight=0）
  ③ 累积 accum_groups 个题组再更新一次，降低题间噪声

用法：
    # 冒烟：8 道题，2 次更新
    python train_grpo2.py --n 8 --k 4 --accum_groups 4 --steps 2

    # 正式：500 道题 × k=8，4 组累积
    python train_grpo2.py --n 500 --k 8 --accum_groups 4 --lr 1e-6 \
        --kl_weight 0 --out /root/autodl-tmp/out_grpo4
"""

import argparse
import json
import os
import re

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/autodl-tmp/out_base_sft"
DATA = "/root/autodl-tmp/sft_train_clean.jsonl"


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


def reward_simple(text, gold):
    """正确 = 1，其他 = 0（组内 z-score 后与 +1.2/−0.3 等价）。"""
    return 1.0 if same(extract(text), gold) else 0.0


def seq_logprob(model, ids, temp=1.0):
    """返回与 ids 中"下一个 token"对齐的 log 概率数组（长度 seq-1）。

    temp: 与采样时一致的温度。采样用 T，打分也要用 T，否则分布不一致 → 梯度有偏。
    """
    out = model(ids).logits[:, :-1, :].float() / temp
    targets = ids[:, 1:]
    lp = torch.log_softmax(out, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return lp


def truncate_at_eos(row_ids, prompt_len, eos_id):
    """把生成部分截断到第一个 EOS（含 EOS），去掉尾部 padding。

    row_ids 是 Python 列表（来自 .tolist()），所以用 list.index 找 EOS。
    """
    gen = row_ids[prompt_len:]
    if eos_id in gen:
        end = gen.index(eos_id)
    else:
        end = len(gen) - 1
    return row_ids[: prompt_len + end + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--accum_groups", type=int, default=4, help="累积几个题组更新一次")
    ap.add_argument("--epochs", type=int, default=1, help="数据重复几轮")
    ap.add_argument("--steps", type=int, default=0, help="最多多少次更新（0=不限制）")
    ap.add_argument("--save_at", type=int, default=0,
                    help="在这一更新次数时额外保存一个中间 checkpoint（0=不保存）")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl_weight", type=float, default=0.0)
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")

    ref = None
    if args.kl_weight > 0:
        ref = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16
        ).to("cuda")
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj["messages"][0]["content"]
            a = obj["messages"][1]["content"]
            m = re.search(r"####\s*(-?[\d,\.]+)", a)
            if m:
                rows.append({"q": q, "gold": m.group(1).replace(",", "")})
            if len(rows) >= args.n:
                break

    print(
        f"题数: {len(rows)}, k={args.k}, accum_groups={args.accum_groups}, "
        f"epochs={args.epochs}, lr={args.lr}, kl_weight={args.kl_weight}"
    )
    print("=" * 68)

    update = 0
    pending = 0
    n_seen = n_mixed = n_allright = n_allwrong = 0
    opt.zero_grad()

    for r in rows * args.epochs:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["q"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        inputs = tok(prompt, return_tensors="pt").to("cuda")

        policy.eval()
        with torch.no_grad():
            out = policy.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=True,
                temperature=args.temp,
                top_p=0.95,
                num_return_sequences=args.k,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)

        rewards = torch.tensor(
            [reward_simple(t, r["gold"]) for t in texts], device="cuda"
        )
        n_seen += 1

        if rewards.std() < 1e-6:
            if rewards.mean() > 0.5:
                n_allright += 1
            else:
                n_allwrong += 1
            continue

        n_mixed += 1
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        policy.train()
        pk = len(p_ids)
        for i in range(args.k):
            ids_row = truncate_at_eos(out[i].tolist(), pk, eos_id)
            ids = torch.tensor([ids_row], device="cuda")

            lp = seq_logprob(policy, ids, args.temp)[0, pk - 1:]
            pg = -(adv[i] * lp.mean())

            if ref is not None:
                with torch.no_grad():
                    lp_ref = seq_logprob(ref, ids, args.temp)[0, pk - 1:]
                d = lp_ref - lp
                k3 = (torch.expm1(d) - d).mean()
                pg = pg + args.kl_weight * k3

            (pg / (args.accum_groups * args.k)).backward()

        pending += 1

        if pending % args.accum_groups == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            update += 1
            if update % 5 == 0:
                print(
                    f"更新 {update:4d}  已见题 {n_seen}  混合 {n_mixed} "
                    f"(全对 {n_allright} / 全错 {n_allwrong})  "
                    f"reward均值 {rewards.mean().item():.2f}"
                )
            if args.save_at and update == args.save_at and args.out:
                mid = args.out + f"_u{args.save_at}"
                policy.save_pretrained(mid)
                tok.save_pretrained(mid)
                print(f"★ 中间 checkpoint 已保存: {mid}（更新 {update} 次，见题 {n_seen}）")
            if args.steps and update >= args.steps:
                break

    # 收尾：不足 accum_groups 的残余
    if pending % args.accum_groups != 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        update += 1

    print("=" * 68)
    print(
        f"完成：更新 {update} 次，见题 {n_seen}，混合 {n_mixed}，"
        f"全对 {n_allright}，全错 {n_allwrong}"
    )
    print(f"混合组比例：{n_mixed/max(n_seen,1)*100:.1f}%")

    if args.out:
        policy.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print("已保存到", args.out)


if __name__ == "__main__":
    main()
