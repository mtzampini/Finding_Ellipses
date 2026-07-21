import numpy as np
from kippenhahn_symbolic import coefficient_extractor

def matrix_to_params_batch(H_batch, K_batch):
    return (
        H_batch[:,0,0].real, H_batch[:,1,1].real, H_batch[:,2,2].real,
        H_batch[:,0,1].real, H_batch[:,0,1].imag,
        H_batch[:,0,2].real, H_batch[:,0,2].imag,
        H_batch[:,1,2].real, H_batch[:,1,2].imag,
        K_batch[:,0,0].real, K_batch[:,1,1].real, K_batch[:,2,2].real,
        K_batch[:,0,1].real, K_batch[:,0,1].imag,
        K_batch[:,0,2].real, K_batch[:,0,2].imag,
        K_batch[:,1,2].real, K_batch[:,1,2].imag,
    )

def extract_H_K_batch(A_batch):
    A_star = np.conj(np.transpose(A_batch, axes=(0, 2, 1)))
    H_batch = (A_batch + A_star) / 2
    K_batch = (A_batch - A_star) / (2j)
    return H_batch, K_batch

def normalize_batch(A_batch, eps=1e-14):
    norms = np.linalg.norm(A_batch, ord='fro', axis=(1, 2))  
    norms = np.maximum(norms, eps)  
    return A_batch / norms[:, None, None]

def feature_extractor(A_batch):
    A_norm = normalize_batch(A_batch)
    H_batch, K_batch = extract_H_K_batch(A_norm)
    params = matrix_to_params_batch(H_batch, K_batch)

    coeffs = coefficient_extractor(*params)
    n = A_batch.shape[0]
    coeffs = [np.broadcast_to(c, (n,)) for c in coeffs]

    return np.stack(coeffs, axis=1)[:, :9]







