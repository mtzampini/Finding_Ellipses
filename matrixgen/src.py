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

# case 2: reducible matrices

def random_reducible_matrix(inside, scale=2.0):
    lam1 = np.random.randn() + 1j*np.random.randn()
    lam2 = np.random.randn() + 1j*np.random.randn()
    p = (np.random.rand() + 1j*np.random.randn()) * scale
    B = np.array([[lam1, p], [0, lam2]])

    center = (lam1 + lam2) / 2
    b = abs(p/2)
    c = abs(lam1 - lam2) / 2
    a = np.sqrt(b**2 + c**2)
    uhat = (lam2 - lam1) / (2*c) if c < 1e-6 else 1
    uper = 1j * uhat 

    r = np.random.rand(0, 0.9) if inside else np.random.rand(1, 2.5)
    t = np.random.uniform(0, 2*np.pi)

    lam0 = center + r * (a*np.cos(t)*uhat + b*np.sin(t)*uper)
    block = np.zeros((3, 3), dtype=complex)
    block[:2, :2] = B
    block[2, 2] = lam0
    U = random_orthonormal_basis(3)
    A = U @ block @ U.conj().T
    return A
