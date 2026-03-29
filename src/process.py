import pandas as pd
import config

from sklearn.model_selection import train_test_split


def process():
    print("开始处理数据")
    # 1. 读取数据
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt', sep='\t', header=None, usecols=[0, 1], names=['en', 'zh'],
                encoding="utf-8").dropna()

    # 2. 划分数据集
    train_df,test_df = train_test_split(df,test_size=0.2)


    # 3. 构建词表

    


