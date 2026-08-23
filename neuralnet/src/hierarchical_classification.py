from model import MatrixClassifierMLP
import numpy as np
import torch
from dataset import MatrixDataset
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import plotting_utils


dataset_file = np.load('numerical-experiments/neuralnet/data/dataset_arrays.npz')
X_val=dataset_file["X_val"]
y_val=dataset_file["y_val"]
val_dataset = MatrixDataset(X_val, y_val)

model_A = MatrixClassifierMLP(n_classes=2)
model_A.load_state_dict(torch.load('numerical-experiments/neuralnet/checkpoints/node_A.pt'))
model_A.eval()

model_B = MatrixClassifierMLP(n_classes=2)
model_B.load_state_dict(torch.load('numerical-experiments/neuralnet/checkpoints/node_B.pt'))
model_B.eval()

model_C = MatrixClassifierMLP(n_classes=2)
model_C.load_state_dict(torch.load('numerical-experiments/neuralnet/checkpoints/node_C.pt'))
model_C.eval()

final_predictions = np.zeros(len(y_val))
indices = np.arange(len(y_val))

output_A = model_A(val_dataset.X)
y_pred_A = torch.argmax(output_A, dim=1).cpu().numpy()
final_predictions[indices[y_pred_A == 1]] = 1
continue_mask = (y_pred_A == 0)
val_dataset.X = val_dataset.X[continue_mask]
indices = indices[continue_mask]

output_B = model_B(val_dataset.X)
y_pred_B = torch.argmax(output_B, dim=1).cpu().numpy()
final_predictions[indices[y_pred_B == 1]] = 2
continue_mask = (y_pred_B == 0)
val_dataset.X = val_dataset.X[continue_mask]
indices = indices[continue_mask]

output_C = model_C(val_dataset.X)
y_pred_C = torch.argmax(output_C, dim=1).cpu().numpy()
final_predictions[indices[y_pred_C == 1]] = 3
final_predictions[indices[y_pred_C == 0]] = 4

print(np.unique(final_predictions, return_counts=True))
print(classification_report(y_val, final_predictions))
conf_matrix = confusion_matrix(y_val, final_predictions)
print(conf_matrix)

class_names = ['Normal', 'Reducible', 'Flat', 'Generic']
plotting_utils.save_confusion_matrix(conf_matrix, class_names, 'Hierarchical Model Confusion Matrix', 'numerical-experiments/neuralnet/results/hierarchical_confusion_matrix.png')