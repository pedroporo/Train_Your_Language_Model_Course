from torch.utils.data import Dataset
import torch


class GPTDataset(Dataset):
    def __init__(self, data, block_size, split='train', train_split=0.9):
        self.data = data
        self.block_size = block_size
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = int(train_split * len(data))
        else:
            self.start_idx = int(train_split * len(data))
            self.end_idx = len(data)
            
        self.length = self.end_idx - self.start_idx - block_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        x = torch.tensor(self.data[actual_idx:actual_idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[actual_idx + 1:actual_idx + self.block_size + 1], dtype=torch.long)
        return x, y