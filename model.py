from dataclasses import dataclass

import torch
from torch import nn

# GPTConfig 是一个数据类，用于存储 GPT 模型的配置参数。
@dataclass
class GPTConfig:
    vocab_size: int = 8192 # 词表的大小
    block_size: int = 256 # 输入样本（模型支持）的最大长度
    n_embd: int = 256 # 每个token用多少个si

# 定义神经网络模块
class TokenPositionEmbedding(nn.Module):
    # 构建2张可学习的向量表
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size
        # 表示一个token，该token的表示向量+其所处位置的位置向量
        # token向量表，每个token保存可学习的向量
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )
        # 位置向量表，每个位置保存一个可学习的向量，不需要知道位置上是什么token
        self.position_embedding = nn.Embedding(
            config.block_size,config.n_embd
        )

    def forward(self,idx): # self表示访问自己的模块内容
        batch_size, sequence_length = idx.shape

        if sequence_length > self.block_size:
            raise ValueError("输入长度超过模型支持的最大长度。")

        # 为输入中的每个位置生成一个编号，并把它放到与输入相同的设备上。
        positions = torch.arange(
            sequence_length, device=idx.device
        )

        # 生成token向量
        token_vectors = self.token_embedding(idx) 
        # 生成位置向量
        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors


if __name__ == "__main__":
    from dataset import get_batch, tokenizer

    # 创建配置(第8行)
    config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    # 根据配置创建嵌入层
    embedding = TokenPositionEmbedding(config) 

    x, y = get_batch("train", batch_size=2, block_size=8)
    # 把编号送入嵌入模块
    vectors = embedding(x)

    print("输入编号形状：", x.shape)
    print("输出向量形状：", vectors.shape)
    print("输出数据类型：", vectors.dtype)


