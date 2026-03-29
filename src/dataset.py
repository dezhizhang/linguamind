import pandas as pd
import torch
import config
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class LinguaMindDataset(Dataset):

    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['en'], dtype=torch.long)
        return input_tensor, target_tensor

def collate_fn(batch):
    input_tensor = [item[0] for item in batch]
    target_tensor = [item[0] for item in batch]

    input_tensor = pad_sequence(input_tensor, batch_first=True,padding_value=0)
    target_tensor =pad_sequence(target_tensor, batch_first=True,padding_value=0)

    return input_tensor, target_tensor




def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = LinguaMindDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,collate_fn=collate_fn)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor,target_tensor in train_dataloader:
        print(input_tensor.shape,target_tensor.shape)
        break

