# 打乱清洗后的数据，划分为训练集、验证集和测试集。
import json
import random
from pathlib import Path

project_dir = Path(__file__).resolve().parent
processed_dir = project_dir / "data" / "processed"
input_file = processed_dir / "cleaned.jsonl"

records = []

with input_file.open("r", encoding="utf-8") as source:
    for line in source:
        records.append(json.loads(line))

# 打乱列表，减少原始排列顺序对划分的影响
random.Random(42).shuffle(records) 

train_end = int(len(records) * 0.90)
val_end = int(len(records) * 0.95)

splits = {
    "train": records[:train_end],
    "val": records[train_end:val_end],
    "test": records[val_end:],
}

for name, samples in splits.items():
    # 将清理后的文本数据集划分为训练集、验证集和测试集
    output_file = processed_dir / f"{name}.jsonl"

    with output_file.open("w", encoding="utf-8") as target:
        for sample in samples:
            # 逐条写入各个集中的sample
            target.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"{name}：{len(samples)} 条，保存到 {output_file}")


