"""清洗 GSM8K 训练数据：去掉答案里的 <<计算器标注>>。

"48/2 = <<48/2=24>>24 clips"  →  "48/2 = 24 clips"
"""

import json
import re

SRC = "/root/autodl-tmp/sft_train.jsonl"
DST = "/root/autodl-tmp/sft_train_clean.jsonl"

# 匹配 <<...>>，包括里面的等号、加减乘除
CALC = re.compile(r"<<[^>]*>>")


def clean(text):
    text = CALC.sub("", text)              # 删掉标注
    text = re.sub(r"[ \t]{2,}", " ", text)  # 合并多余空格
    text = re.sub(r" +\n", "\n", text)      # 行尾空格
    return text.strip()


def main():
    n = 0
    changed = 0
    with open(SRC, encoding="utf-8") as f, open(DST, "w", encoding="utf-8") as g:
        for line in f:
            obj = json.loads(line)
            msgs = obj["messages"]
            before = msgs[-1]["content"]
            after = clean(before)
            if after != before:
                changed += 1
            msgs[-1]["content"] = after
            g.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"清洗完成: {n} 条，其中 {changed} 条被修改")
    print(f"输出: {DST}")

    # 打印第一条对照
    with open(SRC, encoding="utf-8") as f:
        old = json.loads(f.readline())["messages"][-1]["content"]
    with open(DST, encoding="utf-8") as f:
        new = json.loads(f.readline())["messages"][-1]["content"]
    print("\n--- 清洗前 ---")
    print(old)
    print("\n--- 清洗后 ---")
    print(new)


if __name__ == "__main__":
    main()
