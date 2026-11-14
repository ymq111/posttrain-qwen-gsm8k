"""阶段3：SFT 训练脚本（全参微调，不依赖 trl / peft）。

用法：
    python train_sft.py --limit 100     # 小规模验证管道
    python train_sft.py                 # 全量 7473 条
"""

import argparse
import os

# 强制离线，直接用数据盘里的缓存
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LEN = 576
DATA = "/root/autodl-tmp/sft_train.jsonl"


def build_features(tok, messages, max_len):
    """一条对话 → input_ids + labels。

    关键：system 和 user 部分标成 -100（不算 loss），只学 assistant 的回答。
    """
    prompt = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    answer = messages[-1]["content"] + tok.eos_token

    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]

    input_ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条，0 表示全量")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--data", default=DATA, help="训练数据 jsonl 路径")
    ap.add_argument("--model", default=MODEL, help="模型名或本地路径")
    ap.add_argument("--out", default="/root/autodl-tmp/out_sft")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    n_param = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"模型参数量: {n_param:.2f}B")

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"训练样本: {len(ds)} 条")

    ds = ds.map(
        lambda ex: build_features(tok, ex["messages"], MAX_LEN),
        remove_columns=ds.column_names,
        desc="tokenize",
    )

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("已保存到", args.out)


if __name__ == "__main__":
    main()
