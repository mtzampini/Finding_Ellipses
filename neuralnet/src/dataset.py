import torch
from torch.utils.data import Dataset

class MatrixDataset(Dataset):
    def __init__(self, X, y, apply_shift=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        if apply_shift:
            self.y = self.y - 1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    