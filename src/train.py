import torch
import config
import time
from dataset import get_dataloader
from tokenizer import EnglishTokenizer, ChineseTokenizer
from model import TranslatorModel
from torch.utils.tensorboard import SummaryWriter

def train():
    # 1. 获取训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 获取训练数据
    dataloader = get_dataloader()

    # 3. 分词器
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')

    # 4. 构建模型
    model = TranslatorModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.padding_idx,
        en_tokenizer.padding_idx
    ).to(device)

    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d-%H-%M-%S"))

    best_loss = float('inf')




