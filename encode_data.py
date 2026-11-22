# 用分词器转换成token编号，并保存为.npy文件
# 注意!!! 转化为的npy文件会无视原来的json结构，但是还保存<eos>,
# 后续处理片段会跨越不止一个原来的json，产生无关上下文
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# 加载已经训练好的分词器
project_dir = Path(__file__).resolve().parent
processed_dir = project_dir / "data" / "processed"
tokenizer_file = project_dir / "tokenizer" / "tokenizer.json"

tokenizer = Tokenizer.from_file(str(tokenizer_file))
eos_id = tokenizer.token_to_id("<eos>")

if eos_id is None:
    raise ValueError("分词器中缺少 <eos>，请先检查分词器。")


# 定义编码函数，逐条收集编号
def encode_split(name):
    input_file = processed_dir / f"{name}.jsonl"
    output_file = processed_dir / f"{name}_tokens.npy"

    token_ids = []
    offsets = [0]  # 用于记录每条文本的起始位置
    record_count = 0

    with input_file.open("r", encoding="utf-8") as source:
        for line in source:
            sample = json.loads(line)
            text = sample["text"]

            ids = tokenizer.encode(
                text, 
                add_special_tokens=False # 不让分词器的后处理自动添加标记
            ).ids

            token_ids.extend(ids) # 将token编号添加到列表中
            token_ids.append(eos_id) # 添加句子结束符
            # 每一条文本的结束位置就是当前token_ids的长度
            offsets.append(len(token_ids)) 
            record_count += 1

    tokens = np.array(token_ids, dtype=np.int32)
    np.save(output_file, tokens, allow_pickle=False)
    offsets_file = processed_dir / f"{name}_offsets.npy"
    offsets_array = np.array(offsets, dtype=np.int64)
    np.save(offsets_file, offsets_array, allow_pickle=False)

    print(f"记录边界保存到：{offsets_file}")

    print(f"{name}：处理了 {record_count} 条文本")
    print(f"token 总数：{tokens.size}")
    print(f"保存位置：{output_file}")

for split_name in ["train", "val", "test"]:
    encode_split(split_name)

