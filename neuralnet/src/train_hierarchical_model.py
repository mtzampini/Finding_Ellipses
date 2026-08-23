from hierarchical_data import (filter_node_A, filter_node_B, filter_node_C)
from dataset import MatrixDataset
from model import MatrixClassifierMLP, EarlyStopping
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import plotting_utils

np.random.seed(42)
torch.manual_seed(42)

dataset_file = np.load('numerical-experiments/neuralnet/data/dataset_arrays.npz')
X_train = dataset_file['X_train']
y_train=dataset_file["y_train"]
X_val=dataset_file["X_val"]
y_val=dataset_file["y_val"]

# node A: Normal vs Not-Normal
X_train_A = X_train
y_train_A = filter_node_A(y_train)
X_val_A = X_val
y_val_A = filter_node_A(y_val)

train_dataset_A = MatrixDataset(X_train_A, y_train_A, apply_shift=False)
val_dataset_A = MatrixDataset(X_val_A, y_val_A, apply_shift=False)

train_loader = DataLoader(dataset=train_dataset_A, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset_A, batch_size=32, shuffle=False)

_, counts = np.unique(train_dataset_A.y.numpy(), return_counts=True)
counts = torch.tensor(counts, dtype=torch.float32)
weights = counts.sum() / (len(counts) * counts)
criterion = nn.CrossEntropyLoss(weight=weights)
model_A = MatrixClassifierMLP(n_classes=2)
optimizer = torch.optim.Adam(model_A.parameters(), lr=1e-3)
earlystopping = EarlyStopping(patience=10, min_delta=0.001)


epochs = 100
train_losses_A = []
val_losses_A = []
accuracy_A = []

for epoch in range(epochs):
    model_A.train()
    train_epoch_losses = []
    epoch_accuracy = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model_A(X_batch)
        # metrics
        loss = criterion(output, y_batch)
        train_epoch_losses.append(loss.item())
        # backprop
        loss.backward()
        optimizer.step()

    train_losses_A.append(np.mean(train_epoch_losses))

    model_A.eval()
    val_epoch_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model_A(X_batch)
            loss = criterion(output, y_batch)
            val_epoch_losses.append(loss.item())
            predictions = torch.argmax(output, dim=1)
            batch_accuracy = (predictions == y_batch).float().mean()
            epoch_accuracy.append(batch_accuracy.item())

    accuracy_A.append(np.mean(epoch_accuracy))
    avg_val_loss = np.mean(val_epoch_losses)
    val_losses_A.append(avg_val_loss)

    earlystopping(avg_val_loss, model_A)
    if earlystopping.early_stop:
        print(f'Interrupted training at epoch {epoch+1}')
        break

model_A.load_state_dict(earlystopping.best_weights)
torch.save(model_A.state_dict(), 'numerical-experiments/neuralnet/checkpoints/node_A.pt')

plotting_utils.save_loss_curve(train_losses_A, val_losses_A, 'Node A Loss (Normal vs Non-Normal)', 'numerical-experiments/neuralnet/results/node_A_loss_curve.png')
plotting_utils.save_accuracy_curve(None, accuracy_A, 'Node A Accuracy', 'numerical-experiments/neuralnet/results/node_A_accuracy_curve.png')

# node B: Reducible vs Irreducible
X_train_B, y_train_B = filter_node_B(X_train, y_train)
X_val_B, y_val_B = filter_node_B(X_val, y_val)

train_dataset_B = MatrixDataset(X_train_B, y_train_B, apply_shift=False)
val_dataset_B = MatrixDataset(X_val_B, y_val_B, apply_shift=False)

train_loader = DataLoader(dataset=train_dataset_B, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset_B, batch_size=32, shuffle=False)

counts = np.bincount(train_dataset_B.y.numpy(), minlength=2)
counts = torch.tensor(counts, dtype=torch.float32)
weights = counts.sum() / (len(counts) * counts)
criterion = nn.CrossEntropyLoss(weight=weights)
model_B = MatrixClassifierMLP(n_classes=2)
optimizer = torch.optim.Adam(model_B.parameters(), lr=1e-3)
earlystopping = EarlyStopping(patience=10, min_delta=0.001)


epochs = 100
train_losses_B = []
val_losses_B = []
accuracy_B = []

for epoch in range(epochs):
    model_B.train()
    train_epoch_losses = []
    epoch_accuracy = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model_B(X_batch)
        # metrics
        loss = criterion(output, y_batch)
        train_epoch_losses.append(loss.item())
        # backprop
        loss.backward()
        optimizer.step()

    train_losses_B.append(np.mean(train_epoch_losses))

    model_B.eval()
    val_epoch_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model_B(X_batch)
            loss = criterion(output, y_batch)
            val_epoch_losses.append(loss.item())
            predictions = torch.argmax(output, dim=1)
            batch_accuracy = (predictions == y_batch).float().mean()
            epoch_accuracy.append(batch_accuracy.item())

    accuracy_B.append(np.mean(epoch_accuracy))
    avg_val_loss = np.mean(val_epoch_losses)
    val_losses_B.append(avg_val_loss)

    earlystopping(avg_val_loss, model_B)
    if earlystopping.early_stop:
        print(f'Interrupted training at epoch {epoch+1}')
        break

model_B.load_state_dict(earlystopping.best_weights)
torch.save(model_B.state_dict(), 'numerical-experiments/neuralnet/checkpoints/node_B.pt')

plotting_utils.save_loss_curve(train_losses_B, val_losses_B, 'Node B Loss (Reducible vs Irreducible)', 'numerical-experiments/neuralnet/results/node_B_loss_curve.png')
plotting_utils.save_accuracy_curve(None, accuracy_B, 'Node B Accuracy', 'numerical-experiments/neuralnet/results/node_B_accuracy_curve.png')

# node C: flat boundary vs generic
X_train_C, y_train_C = filter_node_C(X_train, y_train)
X_val_C, y_val_C = filter_node_C(X_val, y_val)

train_dataset_C = MatrixDataset(X_train_C, y_train_C, apply_shift=False)
val_dataset_C = MatrixDataset(X_val_C, y_val_C, apply_shift=False)

train_loader = DataLoader(dataset=train_dataset_C, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset_C, batch_size=32, shuffle=False)

counts = np.bincount(train_dataset_B.y.numpy(), minlength=2)
counts = torch.tensor(counts, dtype=torch.float32)
weights = counts.sum() / (len(counts) * counts)
criterion = nn.CrossEntropyLoss(weight=weights)
model_C = MatrixClassifierMLP(n_classes=2)
optimizer = torch.optim.Adam(model_C.parameters(), lr=1e-3)
earlystopping = EarlyStopping(patience=10, min_delta=0.001)


epochs = 100
train_losses_C = []
val_losses_C = []
accuracy_C = []

for epoch in range(epochs):
    model_C.train()
    train_epoch_losses = []
    epoch_accuracy = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model_C(X_batch)
        # metrics
        loss = criterion(output, y_batch)
        train_epoch_losses.append(loss.item())
        # backprop
        loss.backward()
        optimizer.step()

    train_losses_C.append(np.mean(train_epoch_losses))

    model_C.eval()
    val_epoch_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model_C(X_batch)
            loss = criterion(output, y_batch)
            val_epoch_losses.append(loss.item())
            predictions = torch.argmax(output, dim=1)
            batch_accuracy = (predictions == y_batch).float().mean()
            epoch_accuracy.append(batch_accuracy.item())

    accuracy_C.append(np.mean(epoch_accuracy))
    avg_val_loss = np.mean(val_epoch_losses)
    val_losses_C.append(avg_val_loss)

    earlystopping(avg_val_loss, model_C)
    if earlystopping.early_stop:
        print(f'Interrupted training at epoch {epoch+1}')
        break

model_C.load_state_dict(earlystopping.best_weights)
torch.save(model_C.state_dict(), 'numerical-experiments/neuralnet/checkpoints/node_C.pt')

plotting_utils.save_loss_curve(train_losses_C, val_losses_C, 'Node C Loss (Flat vs Generic)', 'numerical-experiments/neuralnet/results/node_C_loss_curve.png')
plotting_utils.save_accuracy_curve(None, accuracy_C, 'Node C Accuracy', 'numerical-experiments/neuralnet/results/node_C_accuracy_curve.png')