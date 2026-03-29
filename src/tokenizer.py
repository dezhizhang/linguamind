from tqdm import tqdm


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
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]


    @classmethod
    def tokenize(text) -> list[str]:
        """简单按空格切分，可按需改进"""
        return text.strip().split()

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """构建词表并保存到文件"""
        vocab_set = set()
        for sentence in tqdm(sentences, desc="building vocab"):
            vocab_set.update(cls.tokenize(sentence))  # 注意这里实例化以调用 tokenize

        # 构建最终词表：特殊字符 + sorted vocab
        vocab_list = [cls.unk_token, cls.pad_token, cls.sos_token, cls.eos_token] + sorted(vocab_set)

        # 写入文件
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list))


    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    """中文分词器"""
    @classmethod
    def tokenize(cls,text) -> list[str]:
        return list(text)


class EnglishTokenizer(BaseTokenizer):
    """英文分词器"""
    tokenizer = TreebankWordTokenizer()

    @classmethod
    def tokenize(cls,text) -> list[str]:
        return cls.tokenizer.tokenize(text)


    def decode(self, indexes):


        # return self.tokenizer.detokenize(text)





if __name__ == "__main__":
    # 测试
   tokenizer = TreebankWordTokenizer()
   word_list = tokenizer.tokenize("hello world")
   print(word_list)



