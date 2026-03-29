import pandas as pd
import config

from sklearn.model_selection import train_test_split
from tokenizer import EnglishTokenizer, ChineseTokenizer


def process():
    print("开始处理数据")
    # 1. 读取数据
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt', sep='\t', header=None, usecols=[0, 1], names=['en', 'zh'],
                     encoding="utf-8").dropna()

    # 2. 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2)

    # 3. 构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.MODELS_DIR / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.MODELS_DIR / 'en_vocab.txt')

    # 4. 构建Tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')

    # 5. 构建训练集
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_special_tokens=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, add_special_tokens=True))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)

    # 7. 构建测试集
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_special_tokens=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_special_tokens=True))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)


if __name__ == "__main__":
    process()
