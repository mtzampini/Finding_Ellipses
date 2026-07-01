import numpy as np


def random_orthonormal_basis(n):
    A = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    d = d / np.abs(d)
    return Q*d

# case 1: normal matrices

def random_eigenvalues(k):
    """
    k = number of distinct eigenvalues
    """
    eigvals = np.random.randn(k) + 1j*np.random.randn(k)
    return eigvals

def random_normal_matrix(n, distinct_eigvals=3):
    eigvals = random_eigenvalues(distinct_eigvals)
    Q = random_orthonormal_basis(n)
    A = Q @ np.diag(eigvals) @ Q.conj().T
    return A

