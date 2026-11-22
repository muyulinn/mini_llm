# 从 data/processed 下的 train_tokens.npy 和 val_tokens.npy 加载编号数组。
# 数组中已没有 JSON 字典结构，每条原始文本末尾保留了 <eos> 标记。
# 当前代码从整条编号序列中随机截取固定长度片段，可能跨越多条原始记录，
# 引入无关上下文，训练时需要注意。

# 1.加载训练集和验证集
from tokenizers import Tokenizer
from pathlib import Path

import numpy as np
import torch

project_dir = Path(__file__).resolve().parent
processed_dir = project_dir / "data" / "processed"

datasets = {
    "train": np.load(
        processed_dir / "train_tokens.npy",
        mmap_mode="r", # 只读方式访问
        allow_pickle=False, # 不允许使用 pickle 反序列化对象
    ),
    "val": np.load(
        processed_dir / "val_tokens.npy",
        mmap_mode="r", # 只读方式访问
        allow_pickle=False, # 不允许使用 pickle 反序列化对象
    ),
}

record_offsets = {
    "train": np.load(
        processed_dir / "train_offsets.npy",
        allow_pickle=False, # 不允许使用 pickle 反序列化对象
    ),
    "val": np.load(
        processed_dir / "val_offsets.npy",
        allow_pickle=False, # 不允许使用 pickle 反序列化对象
    ),
}

tokenizer = Tokenizer.from_file(
    str(project_dir / "tokenizer" / "tokenizer.json")
)
pad_id = tokenizer.token_to_id("<pad>") # pad用于填充，保证每个batch的长度一致

if pad_id is None:
    raise ValueError("分词器中缺少 <pad>，请先检查分词器。")


# split: "train" 或 "val"
# batch_size: 一次取多少个样本
# block_size: 每个样本多少个token
# device: "cpu" 或 "cuda"
def get_batch(split, batch_size, block_size, device="cpu"):
    tokens = datasets[split]
    offsets = record_offsets[split]
    record_count = len(offsets) - 1

    # 随机选择 batch_size 条记录，一条记录对应一个json
    # record_ids记录的是每条记录开头的索引位置
    record_ids = torch.randint(record_count, (batch_size,))
    
    x = torch.full(
        (batch_size, block_size), pad_id, dtype=torch.long
    ) # 用 pad_id 填充输入序列，保证每个样本长度一致
    y = torch.full(
        (batch_size, block_size), -100, dtype=torch.long
    ) # 用 -100 填充目标序列使样本长度一致，-100 在计算 loss 时会被忽略

    for row, record_id in enumerate(record_ids.tolist()):
        # [start, end) 
        start = int(offsets[record_id])
        end = int(offsets[record_id + 1])

        # 计算最大的有效起点数的索引，间接反映有多少个有效起点数，
        max_shift = max(0, end - start - block_size - 1) 
        # 从 0 - max_shift 中随机选择一个偏移量
        shift = torch.randint(max_shift + 1, (1,)).item()
        # 根据偏移量调整起始位置
        start += shift

        # 可以取到的最大终点索引，保证不会越界
        stop = min(start + block_size + 1, end)
        # piece作为这个片段
        piece = np.array(tokens[start:stop], dtype=np.int64)

        valid_length = len(piece) - 1
        x[row, :valid_length] = torch.from_numpy(piece[:-1])
        y[row, :valid_length] = torch.from_numpy(piece[1:])

    return x.to(device), y.to(device)

if __name__ == "__main__":
    torch.manual_seed(42) # 设置随机种子，保证结果可复现

    x, y = get_batch("train", batch_size=2, block_size=256)

    print("输入形状：", x.shape)
    print("目标形状：", y.shape)
    print("每条样本的有效目标数：", (y != -100).sum(dim=1).tolist())
    print("第一条输入的前12个编号：", x[0, :12].tolist())
    print("第一条目标的前12个编号：", y[0, :12].tolist())