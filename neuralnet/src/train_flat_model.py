from dataset import MatrixDataset
from model import MatrixClassifierMLP, EarlyStopping
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import plotting_utils

np.random.seed(42)
torch.manual_seed(42)

dataset_file = np.load('numerical-experiments/neuralnet/data/dataset_arrays.npz')
X_train = dataset_file['X_train']
y_train=dataset_file["y_train"]
X_val=dataset_file["X_val"]
y_val=dataset_file["y_val"]

train_dataset = MatrixDataset(X_train, y_train)
val_dataset = MatrixDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

model = MatrixClassifierMLP()
_, counts = np.unique(train_dataset.y.numpy(), return_counts=True)
counts = torch.tensor(counts, dtype=torch.float32)
weights = counts.sum() / (len(counts) * counts)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
earlystopping = EarlyStopping(patience=10, min_delta=0.001)

epochs = 100
train_losses = []
val_losses = []
accuracy = []

for epoch in range(epochs):
    model.train()
    train_epoch_losses = []
    epoch_accuracy = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        # metrics
        loss = criterion(output, y_batch)
        train_epoch_losses.append(loss.item())

        # backprop
        loss.backward()
        optimizer.step()

    train_losses.append(np.mean(train_epoch_losses))

    model.eval()
    val_epoch_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_epoch_losses.append(loss.item())
            predictions = torch.argmax(output, dim=1)
            batch_accuracy = (predictions == y_batch).float().mean()
            epoch_accuracy.append(batch_accuracy.item())

    avg_val_loss = np.mean(val_epoch_losses)
    val_losses.append(avg_val_loss)
    accuracy.append(np.mean(epoch_accuracy))

    earlystopping(avg_val_loss, model)
    if earlystopping.early_stop:
        print(f'Interrupted training at epoch {epoch+1}')
        break

model.load_state_dict(earlystopping.best_weights)
torch.save(model.state_dict(), 'numerical-experiments/neuralnet/checkpoints/flat_model.pt')

# confusion matrix
model.eval()
with torch.no_grad():
    output = model(val_dataset.X)
    y_pred = torch.argmax(output, dim=1).cpu().numpy()
    y_true = val_dataset.y.cpu().numpy()
    conf_matrix = confusion_matrix(y_true, y_pred)

report = classification_report(y_true, y_pred, target_names=['1', '2', '3', '4'])
print(report)
print(conf_matrix)

class_names = ['Normal', 'Reducible', 'Flat', 'Generic']
plotting_utils.save_loss_curve(train_losses, val_losses, 'Flat Model Loss', 'numerical-experiments/neuralnet/results/flat_loss_curve.png')
plotting_utils.save_accuracy_curve(None, accuracy, 'Flat Model Accuracy', 'numerical-experiments/neuralnet/results/flat_accuracy_curve.png')
plotting_utils.save_confusion_matrix(conf_matrix, class_names, 'Flat Model Confusion Matrix', 'numerical-experiments/neuralnet/results/flat_confusion_matrix.png')
