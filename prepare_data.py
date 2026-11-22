# 负责读取原始数据，把处理后的文本保存
# 到 data/processed，保留原始文件供以后追溯。

import json
from itertools import islice # 限制文件读取记录条数
from pathlib import Path


project_dir = Path(__file__).resolve().parent
input_file = project_dir / "data" / "raw" / "pretrain_t2t_mini.jsonl"
output_file = project_dir / "data" / "processed" / "cleaned.jsonl"

max_records = 10000  # 最大处理条数
texts = [] # 按顺序保留文本的容器
seen = set() # 去重容器
read_count = 0


with input_file.open("r", encoding="utf-8") as source:
    # 用 islice 限制读取的记录数=max_records
    for line in islice(source, max_records): 
        read_count += 1
        sample = json.loads(line)
        text = sample.get("text")

        # 如果text不是str类型，跳过
        if not isinstance(text, str):
            continue

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        if not 50 <= len(text) <= 2000:
            continue

        # 如果text已经在seen集合中，跳过
        if text in seen:
            continue

        seen.add(text)
        texts.append(text)

with output_file.open("w", encoding="utf-8") as target:
    for text in texts:
        record = {"text": text}
        target.write(json.dumps(record, ensure_ascii=False) + "\n")

print("检查条数：", read_count)
print("保留条数：", len(texts))
print("过滤或去重条数：", read_count - len(texts))
print("保存位置：", output_file)

