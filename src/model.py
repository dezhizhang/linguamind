import torch
from torch import nn

import config


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx
        )

        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True,
        )

    def forward(self, x):
        embed = self.embedding(x)

        output, _ = self.gru(embed)

        lengths = (x != self.embedding.padding_idx).sum(dim=1)

        last_hidden_state = output[torch.arange(output.shape[0]), lengths - 1]

        return last_hidden_state


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_idx
        )

        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,
            out_features=vocab_size,
        )

    def forward(self, x, hidden_0):
        embed = self.embedding(x)

        output, hidden_n = self.gru(embed, hidden_0)

        output = self.linear(output)

        return output, hidden_n


class TranslatorModel(nn.Module):
    """构建训练模型"""

    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size, padding_idx=zh_padding_idx)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size, padding_idx=en_padding_idx)
