import os
from tqdm import tqdm
from nltk.tokenize import TreebankWordTokenizer  # 需要安装 nltk: pip install nltk


# 如果是第一次运行，可能需要下载 nltk 数据
# import nltk
# nltk.download('punkt')

class BaseTokenizer:
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        # 增加安全性检查，防止特殊符号不在词表中导致报错
        self.unk_token_index = self.word2index.get(self.unk_token, 0)
        self.pad_token_index = self.word2index.get(self.pad_token, 0)
        self.sos_token_index = self.word2index.get(self.sos_token, 0)
        self.eos_token_index = self.word2index.get(self.eos_token, 0)

    @classmethod
    def tokenize(cls, text) -> list[str]:
        """默认按空格切分，子类可重写"""
        return text.split()

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    def decode(self, indexes, skip_special_tokens=True):
        """将索引列表还原为文本"""
        tokens = []
        for idx in indexes:
            word = self.index2word.get(idx, self.unk_token)
            if skip_special_tokens and word in [self.pad_token, self.sos_token, self.eos_token, self.unk_token]:
                continue
            tokens.append(word)
        return " ".join(tokens)

    @classmethod
    def build_vocab(cls, sentences, vocab_path, min_freq=1, max_vocab_size=None):
        """构建词表并保存到文件"""
        vocab_count = {}

        # 1. 统计词频
        for sentence in tqdm(sentences, desc="Building Vocab"):
            tokens = cls.tokenize(sentence)
            for token in tokens:
                vocab_count[token] = vocab_count.get(token, 0) + 1

        # 2. 过滤低频词
        filtered_vocab = [token for token, count in vocab_count.items() if count >= min_freq]

        # 3. 排序 (保证每次构建结果一致)
        sorted_vocab = sorted(filtered_vocab)

        # 4. 截断词表大小 (如果设置了 max_vocab_size)
        if max_vocab_size:
            sorted_vocab = sorted_vocab[:max_vocab_size]

        # 5. 构建最终词表：特殊字符 + 排序后的词汇
        # 修复点：这里补全了列表推导式，并进行了排序
        vocab_list = [cls.unk_token, cls.pad_token, cls.sos_token, cls.eos_token] + sorted_vocab

        # 6. 写入文件
        os.makedirs(os.path.dirname(vocab_path) or '.', exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))

        print(f"Vocab saved to {vocab_path}, size: {len(vocab_list)}")

    @classmethod
    def from_vocab(cls, vocab_path):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f.readlines() if line.strip()]
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    """中文分词器 (按字切分)"""

    @classmethod
    def tokenize(cls, text) -> list[str]:
        # 中文通常按字切分，或者使用 jieba 等库
        return list(text.strip())


class EnglishTokenizer(BaseTokenizer):
    """英文分词器"""
    # 类属性，实例化一次即可
    _tokenizer = TreebankWordTokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls._tokenizer.tokenize(text)

    # 继承 BaseTokenizer 的 decode 即可，通常不需要重写，除非需要特殊的 detokenize 逻辑


if __name__ == "__main__":
    # --- 测试数据 ---
    en_sentences = [
        "hello world",
        "this is a test",
        "hello python",
        "nltk is useful"
    ]

    vocab_path = "test_vocab.txt"

    # 1. 构建词表
    print("=== Building Vocab ===")
    EnglishTokenizer.build_vocab(en_sentences, vocab_path)

    # 2. 加载词表
    print("\n=== Loading Tokenizer ===")
    tokenizer = EnglishTokenizer.from_vocab(vocab_path)
    print(f"Vocab Size: {tokenizer.vocab_size}")

    # 3. 测试 Encode
    print("\n=== Encoding ===")
    text = "hello world"
    encoded = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Indices: {encoded}")

    # 4. 测试 Decode
    print("\n=== Decoding ===")
    decoded = tokenizer.decode(encoded)
    print(f"Indices: {encoded}")
    print(f"Text: {decoded}")

    # 5. 测试中文
    print("\n=== Chinese Tokenizer ===")
    cn_tokenizer = ChineseTokenizer(["<unk>", "<pad>", "<sos>", "<eos>", "你", "好", "世", "界"])
    # 注意：实际使用中中文也需要先 build_vocab，这里为了演示直接初始化
    cn_text = "你好"
    cn_encoded = cn_tokenizer.encode(cn_text)
    print(f"Text: {cn_text}")
    print(f"Indices: {cn_encoded}")