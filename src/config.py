from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


# 数据文件目录
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'

# 数据预处理目录
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'

# 数据模型
MODELS_DIR = ROOT_DIR / 'models'

SEQ_LEN = 5
BATCH_SIZE=64
EMBEDDING_DIM=128
HIDDEN_SIZE=256
LEARNING_RATE=1e-3
EPOCHS=1

# 日志目录
LOGS_DIR = ROOT_DIR / 'logs'