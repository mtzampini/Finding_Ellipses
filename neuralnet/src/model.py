import torch.nn as nn
import torch
import copy

class MatrixClassifierMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim1=32, hidden_dim2=16, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, n_classes)
        )

    def forward(self, x):
        return self.net(x)

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            print(f'EarlyStopping counter {self.counter} out of {self.patience}')
            if self.counter > self.patience:
                self.early_stop = True

