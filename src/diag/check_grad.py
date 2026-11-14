"""一分钟诊断：梯度到底有没有在改模型。

测三件事：
  ① tokenizer 对齐（pk 是否等于 inputs 的长度）
  ② 一次更新后，参数是否真的变了
  ③ 一次更新后，正确答案的 logprob 是否上升

如果 ③ 上升 → 梯度正确
如果 ③ 不动但 ② 变了 → advantage 或符号有问题
如果 ② 也不动 → 优化器/grad 没生效

用法：
  python check_grad.py
"""

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
TEMP = 0.8
K = 4


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


def seq_logprob(model, ids, temp):
    out = model(ids).logits[:, :-1, :].float() / temp
    tgt = ids[:, 1:]
    return torch.log_softmax(out, dim=-1).gather(2, tgt.unsqueeze(-1)).squeeze(-1)


def truncate_at_eos(row_ids, prompt_len, eos_id):
    gen = row_ids[prompt_len:]
    end = gen.index(eos_id) if eos_id in gen else len(gen) - 1
    return row_ids[: prompt_len + end + 1]


def main():
    tok = AutoTokenizer.from_pretrained(INST)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        INST, torch_dtype=torch.bfloat16
    ).to("cuda")

    # ---------- ① 对齐检查 ----------
    print("=" * 66)
    print("① tokenizer 对齐检查")
    print("=" * 66)
    q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
    )
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    inputs_full = tok(prompt, return_tensors="pt")
    print(f"  len(p_ids) 不带特殊token : {len(p_ids)}")
    print(f"  inputs['input_ids'] 长度 : {inputs_full['input_ids'].shape[1]}")
    print(f"  → {'✅ 一致' if len(p_ids) == inputs_full['input_ids'].shape[1] else '❌ 不一致！会导致错位'}")
    print(f"  eos_id={eos_id}  pad_id={tok.pad_token_id}")

    # ---------- 找一道混合题 ----------
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            a = o["messages"][1]["content"]
            m = re.search(r"####\s*(-?[\d,\.]+)", a)
            if m:
                rows.append(
                    {"q": o["messages"][0]["content"],
                     "gold": m.group(1).replace(",", "")}
                )
            if len(rows) >= 40:
                break

    target = None
    for r in rows:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["q"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        pi = tok(prompt, return_tensors="pt").to("cuda")
        model.eval()
        with torch.no_grad():
            out = model.generate(**pi, max_new_tokens=384, do_sample=True,
                                 temperature=TEMP, top_p=0.95,
                                 num_return_sequences=K)
        gen = out[:, pi["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        rw = [1.0 if same(extract(t), r["gold"]) else 0.0 for t in texts]
        if 0 < sum(rw) < K:
            target = dict(q=r["q"], gold=r["gold"], prompt=prompt,
                          pi=pi, out=out, texts=texts, rw=rw)
            break

    if target is None:
        print("没找到混合组，换个数据重试")
        return

    print("\n" + "=" * 66)
    print(f"② 用一道混合题做 1 次更新   gold={target['gold']}")
    print("=" * 66)
    for i, (t, r) in enumerate(zip(target["texts"], target["rw"])):
        print(f"  答案{i} reward={r:.0f} 抽取={extract(t)} 长度={len(t)}")

    rw = torch.tensor(target["rw"], device="cuda")
    adv = (rw - rw.mean()) / (rw.std() + 1e-8)
    print(f"  advantage = {[round(a, 3) for a in adv.tolist()]}")

    pk = len(tok(target["prompt"], add_special_tokens=False)["input_ids"])
    out = target["out"]

    # 选一个正确和一个错误的答案，记录 before
    idx_c = next(i for i in range(K) if target["rw"][i] == 1.0)
    idx_w = next(i for i in range(K) if target["rw"][i] == 0.0)

    def get_ids(i):
        row = truncate_at_eos(out[i].tolist(), pk, eos_id)
        return torch.tensor([row], device="cuda")

    ids_c, ids_w = get_ids(idx_c), get_ids(idx_w)

    model.eval()
    with torch.no_grad():
        lp_c_before = seq_logprob(model, ids_c, TEMP)[0, pk - 1:].mean().item()
        lp_w_before = seq_logprob(model, ids_w, TEMP)[0, pk - 1:].mean().item()

    # 记录更新前的参数快照
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    # 一次更新（lr 故意放大到 1e-3，便于观察）
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    opt.zero_grad()
    for i in range(K):
        ids = get_ids(i)
        lp = seq_logprob(model, ids, TEMP)[0, pk - 1:]
        loss = -(adv[i] * lp.mean()) / K
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    # ---------- ② 参数变化 ----------
    total_delta = 0.0
    changed = 0
    for n, p in model.named_parameters():
        d = (p.detach() - before[n]).abs().sum().item()
        total_delta += d
        if d > 0:
            changed += 1
    print(f"\n  参数总变化量 = {total_delta:.6f}")
    print(f"  变化了的参数张量数 = {changed} / {len(before)}")
    print(f"  → {'✅ 参数确实被更新了' if total_delta > 0 else '❌ 参数完全没变（梯度没进去）'}")

    # ---------- ③ logprob 是否上升 ----------
    model.eval()
    with torch.no_grad():
        lp_c_after = seq_logprob(model, ids_c, TEMP)[0, pk - 1:].mean().item()
        lp_w_after = seq_logprob(model, ids_w, TEMP)[0, pk - 1:].mean().item()

    print("\n  正确答案 logprob（平均每token）：")
    print(f"    更新前 {lp_c_before:.6f}  →  更新后 {lp_c_after:.6f}"
          f"  （{lp_c_after - lp_c_before:+.6f}）")
    print("  错误答案 logprob（平均每token）：")
    print(f"    更新前 {lp_w_before:.6f}  →  更新后 {lp_w_after:.6f}"
          f"  （{lp_w_after - lp_w_before:+.6f}）")

    print("\n" + "=" * 66)
    print("判读")
    print("=" * 66)
    if lp_c_after > lp_c_before and lp_w_after < lp_w_before:
        print("  ✅ 正确上升、错误下降 —— 梯度方向正确，实现没问题")
    elif total_delta == 0:
        print("  ❌ 参数没变 —— 梯度没进优化器")
    elif abs(lp_c_after - lp_c_before) < 1e-5:
        print("  ⚠️ 参数变了但 logprob 几乎没动 —— 更新步长太小或 mask 有问题")
    else:
        print("  ⚠️ 方向不符合预期 —— 检查 advantage 符号或 loss 定义")


if __name__ == "__main__":
    main()
