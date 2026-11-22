# 用训练集学习分词规则和词表，保存分词器并验证编码解码。
import json
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

project_dir = Path(__file__).resolve().parent
train_file = project_dir / "data" / "processed" / "train.jsonl"
output_file = project_dir / "tokenizer" / "tokenizer.json"

def train_texts():
    with train_file.open("r", encoding="utf-8") as source:
        for line in source:
            sample = json.loads(line)
            yield sample["text"]

# 分词器选择BPE算法
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 把文本转换成按字节进行处理的表示
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = decoders.ByteLevel() # 解码器

# 设置训练规则，训练然后保存
special_tokens = [
    "<pad>", # 填充符
    "<unk>", # 未知词
    "<bos>", # 句子开头
    "<eos>", # 句子结尾
    "<user>", # 用户输入
    "<assistant>" # 助手输出
]

trainer = trainers.BpeTrainer(
    vocab_size=8192, # 目标词表大小
    min_frequency=2, # 最小词频
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), # 256 种字节表示加入初始表
    special_tokens=special_tokens,
    show_progress=True,
)

# 开始训练并保存分词器
tokenizer.train_from_iterator(train_texts(), trainer=trainer)
# 保存到mini_llm/tokenizer/tokenizer.json
tokenizer.save(str(output_file)) 

# 加载分词器并进行测试
loaded_tokenizer = Tokenizer.from_file(str(output_file))

example = "你好，世界！Hello AI 🌍"
encoded = loaded_tokenizer.encode(example) # 文字转 token 编号
decoded = loaded_tokenizer.decode(encoded.ids) # token 编号转文字

print("实际词表大小：", loaded_tokenizer.get_vocab_size())
print("原始文本：", example)
print("token 编号：", encoded.ids)
print("解码结果：", decoded)
print("是否完整还原：", decoded == example)
print("保存位置：", output_file)
