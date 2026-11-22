# 查看原始数据样本，统计文本数量
# 和长度，了解数据的格式与基本情况。
import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent
data_file = project_dir / "data" / "raw" / "pretrain_t2t_mini.jsonl"

with data_file.open("r", encoding="utf-8") as file:
    for number, line in enumerate(file, start=1):
        sample = json.loads(line)
        text = sample["text"]

        print(f"\n第 {number} 条数据")
        print("字符数：", len(text))
        print("前200个字符：")
        print(text[:200])

        if number == 3:
            break

lengths = []
empty_count = 0

with data_file.open("r", encoding="utf-8") as file:
    for number, line in enumerate(file, start=1):
        sample = json.loads(line)
        text = sample["text"].strip()

        if text:
            lengths.append(len(text))
        else:
            empty_count += 1

        if number == 10000:
            break

print("\n前10000条数据的统计：")
print("已检查条数：", number)
print("非空文本数：", len(lengths))
print("空文本数：", empty_count)
print("最短文本字符数：", min(lengths))
print("最长文本字符数：", max(lengths))
print("平均文本字符数：", round(sum(lengths) / len(lengths), 1))