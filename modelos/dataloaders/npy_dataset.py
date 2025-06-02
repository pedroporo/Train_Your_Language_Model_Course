from torch.utils.data import Dataset
import torch

class NPYDataset(Dataset):
    def __init__(self, data, block_size, split='train'):
        self.data = data
        self.block_size = block_size
        self.split = split
        split_index = int(0.9 * len(data))
        if split == 'train':
            self.start = 0
            self.end = split_index
        else:
            self.start = split_index
            self.end = len(data)
        self.length = self.end - self.start - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.start + idx
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)