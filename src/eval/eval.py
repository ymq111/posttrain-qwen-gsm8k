"""阶段4：在 300 道题上评测模型，输出准确率。

用法：
    python eval.py --model Qwen/Qwen2.5-0.5B-Instruct          # 测 base
    python eval.py --model /root/autodl-tmp/out_sft            # 测 SFT 后
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


def extract(text):
    """宽松抽答案：优先 #### 后面的数字；没有就退回"最后一个数字"。"""
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def extract_strict(text):
    """严格抽答案：只认 #### 后面的数字，没有就算 None。"""
    m = re.search(r"####\s*(-?[\d,\.]+)", text)
    return None if not m else m.group(1).replace(",", "").rstrip(".")


def same(pred, gold):
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="", help="LoRA 的基座路径（测 LoRA 时必填）")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", default="", help="追加到问题后面的提示词")
    ap.add_argument("--show", type=int, default=3, help="打印前 N 条示例")
    args = ap.parse_args()

    base = args.base or args.model
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"          # 批量生成必须左填充
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    if args.base:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model)
        model = model.merge_and_unload()
        print("已加载 LoRA 补丁:", args.model)
    model = model.to("cuda")
    model.eval()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    correct, done = 0, 0
    strict_correct, has_hash, shown = 0, 0, 0
    total_chars = 0

    for i in range(0, len(rows), args.batch):
        chunk = rows[i : i + args.batch]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["question"] + args.extra}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in chunk
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
            )

        gen = out[:, inputs["input_ids"].shape[1] :]
        texts = tok.batch_decode(gen, skip_special_tokens=True)

        for r, t in zip(chunk, texts):
            pred = extract(t)
            ok = same(pred, r["gold"])
            correct += int(ok)

            spred = extract_strict(t)
            if spred is not None:
                has_hash += 1
            strict_correct += int(same(spred, r["gold"]))

            total_chars += len(t)
            done += 1
            if shown < args.show:
                shown += 1
                print("=" * 60)
                print("问题:", r["question"][:80], "...")
                print("标准:", r["gold"], "| 预测:", pred, "|", "✅" if ok else "❌")
                print("输出:", t[:200].replace("\n", " "))

        print(f"进度 {done}/{len(rows)}  宽松 {correct/done:.3f}  严格 {strict_correct/done:.3f}")

    n = len(rows)
    print("=" * 60)
    print(f"宽松模式（无 #### 时取最后一个数字）: {correct}/{n} = {correct/n*100:.2f}%")
    print(f"严格模式（只认 ####）            : {strict_correct}/{n} = {strict_correct/n*100:.2f}%")
    print(f"输出中含 #### 的比例              : {has_hash}/{n} = {has_hash/n*100:.2f}%")
    print(f"平均输出长度                      : {total_chars/n:.0f} 字符")


if __name__ == "__main__":
    main()
