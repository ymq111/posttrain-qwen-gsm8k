"""过拟合检查：拿少量题反复训练，看训练集准确率能不能涨上去。

这是最经典的 sanity check：
  模型再小，也应该能"背下" 32 道题。
  如果连训练集都涨不上去 → 实现有 bug（梯度、mask、advantage...）
  如果涨上去了但泛化不了 → 数据或预算问题

用法：
  python overfit_test.py --n 32 --steps 40
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

INST = "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA = "/root/autodl-tmp/sft_train_clean.jsonl"


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


def load_rows(n):
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            q = o["messages"][0]["content"]
            a = o["messages"][1]["content"]
            m = re.search(r"####\s*(-?[\d,\.]+)", a)
            if not m:
                continue
            rows.append({"q": q, "gold": m.group(1).replace(",", "").rstrip(".")})
            if len(rows) >= n:
                break
    return rows


def truncate_at_eos(row_ids, prompt_len, eos_id):
    gen = row_ids[prompt_len:]
    end = gen.index(eos_id) if eos_id in gen else len(gen) - 1
    return row_ids[: prompt_len + end + 1]


def seq_logprob(model, ids, temp=1.0):
    out = model(ids).logits[:, :-1, :].float() / temp
    targets = ids[:, 1:]
    return torch.log_softmax(out, dim=-1).gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1)


@torch.no_grad()
def measure(model, tok, rows, max_new=384):
    """测这 rows 上的贪心准确率 + 采样准确率。"""
    model.eval()
    g_ok = s_ok = s_tot = 0
    for i in range(0, len(rows), 4):
        chunk = rows[i : i + 4]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["q"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in chunk
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        texts = tok.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for r, t in zip(chunk, texts):
            g_ok += int(same(extract(t), r["gold"]))

        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=4,
        )
        texts = tok.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for j, r in enumerate(chunk):
            for t in texts[j * 4 : (j + 1) * 4]:
                s_ok += int(same(extract(t), r["gold"]))
                s_tot += 1
    model.train()
    return g_ok / len(rows) * 100, s_ok / max(s_tot, 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=INST)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-5, help="故意比正常大，帮助过拟合")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max_new", type=int, default=384)
    ap.add_argument("--report_every", type=int, default=10)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to("cuda")
    policy.train()

    rows = load_rows(args.n)
    print(f"过拟合测试：{len(rows)} 道题，{args.steps} 步，lr={args.lr}，k={args.k}")
    print(f"（正常训练 lr 是 3e-6，这里故意放大到 {args.lr}，帮助过拟合）")

    g, s = measure(policy, tok, rows, args.max_new)
    print(f"\n[第 0 步]  贪心 {g:.1f}%   采样 {s:.2f}%")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    hist = [(0, g, s)]
    mixed_total = 0

    for step in range(1, args.steps + 1):
        r = rows[(step - 1) % len(rows)]
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
        policy.train()

        rewards = []
        for i in range(args.k):
            t = tok.decode(out[i, p_ids.__len__():], skip_special_tokens=True)
            rewards.append(1.0 if same(extract(t), r["gold"]) else 0.0)

        rt = torch.tensor(rewards, device="cuda")
        if rt.std() >= 1e-6:
            mixed_total += 1
            adv = (rt - rt.mean()) / (rt.std() + 1e-8)

            opt.zero_grad()
            pk = len(p_ids)
            for i in range(args.k):
                ids_row = truncate_at_eos(out[i].tolist(), pk, eos_id)
                ids = torch.tensor([ids_row], device="cuda")
                lp = seq_logprob(policy, ids, args.temp)[0, pk - 1 :]
                loss = -(adv[i] * lp.mean()) / args.k
                loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        if step % args.report_every == 0:
            g, s = measure(policy, tok, rows, args.max_new)
            hist.append((step, g, s))
            print(
                f"[第 {step:3d} 步]  贪心 {g:5.1f}%   采样 {s:5.2f}%   "
                f"（混合组累计 {mixed_total}）"
            )

    print("\n" + "=" * 62)
    print(f"{'步数':<8}{'贪心':>12}{'采样':>12}")
    print("-" * 62)
    for st, g, s in hist:
        print(f"{st:<8}{g:>11.1f}%{s:>11.2f}%")
    print("=" * 62)

    g0, s0 = hist[0][1], hist[0][2]
    g1, s1 = hist[-1][1], hist[-1][2]
    print(f"\n贪心 {g0:.1f}% → {g1:.1f}%（{g1-g0:+.1f}）")
    print(f"采样 {s0:.2f}% → {s1:.2f}%（{s1-s0:+.2f}）")
    print("\n判读：")
    print("  训练集准确率明显上升 → 实现没问题，之前是泛化/预算问题")
    print("  完全不动           → 实现有 bug（梯度、mask、advantage 之一）")


if __name__ == "__main__":
    main()
