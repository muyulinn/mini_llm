# 从零构建小型语言模型

使用 Python 和 PyTorch，逐步理解并实现语言模型的数据准备、
分词器、Transformer、训练、评估和文本生成流程。

这是一个持续开发的学习项目。

## 当前进度

已经完成：

- 预训练文本的检查、基础清洗、完全重复去重。
- 训练集、验证集和测试集划分。
- 训练自己的 Byte-level BPE 分词器，词表大小为 8192。
- 将文本转换为 token 编号，并保存记录边界。
- 在单条记录内部采样，构造输入和下一个 token 预测目标。
- 短样本右侧填充，目标中的 -100 留作损失忽略标记。
- 实现 token 嵌入与可学习的位置嵌入。

尚未完成 Transformer 主体、预训练、对话微调和文本生成。
当前没有训练完成的语言模型权重。

## 文件说明

| 文件 | 作用 |
|---|---|
| inspect_data.py | 查看文本样本和长度统计 |
| prepare_data.py | 清洗、筛选与完全重复去重 |
| split_data.py | 划分训练集、验证集和测试集 |
| train_tokenizer.py | 训练并保存分词器 |
| encode_data.py | 保存 token 编号数组及记录边界 |
| dataset.py | 构造一批输入和预测目标 |
| model.py | 当前实现 token 嵌入和位置嵌入 |
| requirements.txt | 安装源及直接依赖版本 |
| tokenizer/tokenizer.json | 已训练的分词器 |

## 开发环境

- Windows、VS Code
- Python 3.11.2
- PyTorch 2.10.0+cu126
- NumPy 2.4.6
- Tokenizers 0.22.2
- 开发电脑：RTX 3050 Laptop，4GB 显存，约 16GB 内存

## 数据来源

使用 MiniMind 数据集中的 `pretrain_t2t_mini.jsonl`，约 1.24GB。

[数据下载页面](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/pretrain_t2t_mini.jsonl)

数据卡列有 Apache-2.0、CC-BY-NC-2.0 等许可，使用时请查看原始说明。
本仓库不包含原始数据及处理后的数据。

当前实验只处理原始文件前 10000 条记录，清洗后保留 9999 条：

- 训练集：8999 条
- 验证集：500 条
- 测试集：500 条

## 准备与运行

在项目根目录的 PowerShell 终端中创建环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

创建本地输出目录：

```powershell
New-Item -ItemType Directory -Force -Path data\raw,data\processed,tokenizer,checkpoints,logs
```

下载数据并放到：

```text
data/raw/pretrain_t2t_mini.jsonl
```

按顺序运行：

```powershell
.\.venv\Scripts\python.exe inspect_data.py
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe split_data.py
.\.venv\Scripts\python.exe train_tokenizer.py
.\.venv\Scripts\python.exe encode_data.py
.\.venv\Scripts\python.exe dataset.py
.\.venv\Scripts\python.exe model.py
```

数据、检查点和日志目录由本地流程使用，不纳入 Git。
目前模型演示只执行嵌入层的前向计算，不进行参数训练。

## 当前限制

- 实验数据规模较小，顺序截取的前 10000 条不代表完整数据分布。
- 长度筛选和完全重复去重不能保证文本准确或高质量。
- 不跨记录采样，也不能保证记录内部是完整、独立的对话。