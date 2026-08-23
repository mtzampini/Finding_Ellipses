import sys
import os
import numpy as np
from sklearn.metrics import f1_score
import torch
import warnings

np.random.seed(42)
torch.manual_seed(42)

# Suppress sklearn UndefinedMetricWarning for 0 division
warnings.filterwarnings('ignore')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from neuralnet.src.dataset import MatrixDataset
from neuralnet.src.model import MatrixClassifierMLP
from neuralnet.src import plotting_utils

def evaluate_flat_model(X_val, y_val):
    model = MatrixClassifierMLP(n_classes=4)
    model.load_state_dict(torch.load('numerical-experiments/neuralnet/checkpoints/flat_model.pt'))
    model.eval()
    with torch.no_grad():
        output = model(X_val)
        y_pred = torch.argmax(output, dim=1).cpu().numpy() + 1
        
    return f1_score(y_val, y_pred, average=None)

def evaluate_hierarchical_model(X_val, y_val, val_dataset):
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

    # Node A
    output_A = model_A(val_dataset.X)
    y_pred_A = torch.argmax(output_A, dim=1).cpu().numpy()
    final_predictions[indices[y_pred_A == 1]] = 1
    continue_mask = (y_pred_A == 0)
    
    X_node_B = val_dataset.X[continue_mask]
    indices_B = indices[continue_mask]

    # Node B
    if len(X_node_B) > 0:
        output_B = model_B(X_node_B)
        y_pred_B = torch.argmax(output_B, dim=1).cpu().numpy()
        final_predictions[indices_B[y_pred_B == 1]] = 2
        continue_mask_B = (y_pred_B == 0)
        
        X_node_C = X_node_B[continue_mask_B]
        indices_C = indices_B[continue_mask_B]

        # Node C
        if len(X_node_C) > 0:
            output_C = model_C(X_node_C)
            y_pred_C = torch.argmax(output_C, dim=1).cpu().numpy()
            final_predictions[indices_C[y_pred_C == 1]] = 3
            final_predictions[indices_C[y_pred_C == 0]] = 4

    return f1_score(y_val, final_predictions, average=None)

def main():
    dataset_file = np.load('numerical-experiments/neuralnet/data/dataset_arrays.npz')
    X_val = dataset_file["X_val"]
    y_val = dataset_file["y_val"]
    
    X_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_tensor = torch.tensor(y_val, dtype=torch.long)
    val_dataset = MatrixDataset(X_tensor, y_tensor)

    f1_flat = evaluate_flat_model(X_tensor, y_val)
    # the dataset in train_hierarchical_model uses raw X_val, let's pass it
    val_dataset_hier = MatrixDataset(X_tensor, y_tensor)
    f1_hier = evaluate_hierarchical_model(X_tensor, y_val, val_dataset_hier)
    
    class_names = ['Normal', 'Reducible', 'Flat', 'Generic']
    f1_scores_dict = {
        'Flat Model': f1_flat,
        'Hierarchical Model': f1_hier
    }
    
    plotting_utils.save_f1_comparison_bar(class_names, f1_scores_dict, 'F1 Score Comparison', 'numerical-experiments/neuralnet/results/flat_vs_hierarchical_f1.png')

if __name__ == '__main__':
    main()
