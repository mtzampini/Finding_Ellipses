import os
import sys

# Add project root and matrixgen to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
if os.path.join(project_root, 'matrixgen') not in sys.path:
    sys.path.append(os.path.join(project_root, 'matrixgen'))

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
from neuralnet.src.model import MatrixClassifierMLP
from matrixgen.feature_extraction import feature_extractor

model = MatrixClassifierMLP(n_classes=4)
model.load_state_dict(torch.load('numerical-experiments/neuralnet/checkpoints/flat_model.pt'))
model.eval()

with open('numerical-experiments/neuralnet/data/dataset_meta.pkl', 'rb') as f:
    meta = pickle.load(f)
scaler = meta['scaler']

def support_gap(H, K, n_theta=720):
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    gaps = np.empty(n_theta)
    for i, theta in enumerate(thetas):
        M = np.cos(theta)*H + np.sin(theta)*K
        ev = np.linalg.eigvalsh(M)
        gaps[i] = ev[-1] - ev[-2]
    return gaps.min()

def commutator_norm(A):
    C = A @ A.conj().T - A.conj().T @ A
    return np.linalg.norm(C, 'fro') / (np.linalg.norm(A, 'fro')**2)

from matrixgen.generator import gen_case3, gen_case4

res3 = gen_case3()
res4 = gen_case4()
A_prime = res3['matrix']
B_prime = res4['matrix']

t_values = np.linspace(0, 1, 100)
confidences = []
rel_gaps = []
comm_norms = []

for t in t_values:
    M = (1-t)*A_prime + t*B_prime
    H = (M + M.conj().T) / 2
    K = (M - M.conj().T) / (2j)

    gap = support_gap(H, K)
    typical_scale = np.abs(np.linalg.eigvalsh(H)).max() + 1e-9
    rel_gaps.append(gap / typical_scale)
    comm_norms.append(commutator_norm(M))

    coeffs = feature_extractor(np.expand_dims(M, axis=0))[0]
    coeffs_scaled = (coeffs - scaler['mean']) / scaler['std']
    x = torch.tensor(coeffs_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze().numpy()
    confidences.append(probs.max())  # confidenza sulla classe predetta, non su una fissa

confidences = np.array(confidences)
rel_gaps = np.array(rel_gaps)
comm_norms = np.array(comm_norms)

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(t_values, confidences, 'b-', label='Model Confidence (max softmax)')
ax1.set_xlabel('t')
ax1.set_ylabel('Confidence', color='b')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(t_values, rel_gaps, 'r--', label='rel_gap (distance to flat edge)')
ax2.set_ylabel('rel_gap', color='r')
ax2.legend(loc='upper right')

plt.title('Model Confidence vs Geometric Certificate (rel_gap)')
plt.tight_layout()
# ensure the saved figure is in the results dir
plt.savefig('numerical-experiments/neuralnet/results/confidence_vs_certificate.png', dpi=150)