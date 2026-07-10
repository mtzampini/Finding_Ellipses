import numpy as np


def random_orthonormal_basis(n):
    A = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    d = d / np.abs(d)
    return Q*d

def random_hermitian(n, scale=1.0):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) * scale
    return (z + z.conj().T)/2

# case 1: normal matrices

def random_eigenvalues(k):
    """
    k = number of distinct eigenvalues
    """
    eigvals = np.random.randn(k) + 1j*np.random.randn(k)
    return eigvals

def gen_case1(n, distinct_eigvals=3):
    eigvals = random_eigenvalues(distinct_eigvals)
    Q = random_orthonormal_basis(n)
    A = Q @ np.diag(eigvals) @ Q.conj().T
    return A

# case 2: reducible matrices

def gen_case2(inside, scale=2.0):
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

# case 3: irreducible matrix, flat portion of the boundary
def gen_case3(scale=2.0, n_theta=720, flat_tol=2e-3):
    a, b = np.sort(np.random.rand(2))[::-1] * scale
    if a - b < 0.5:
        a = b + 1
    
    U = random_orthonormal_basis(3)
    H = U @ np.diag([a, a, b]) @ U.conj().T
    K = random_hermitian(3, scale)
    A = H + 1j * K

    thetas = np.linspace(0, 2*np.pi, n_theta)
    gaps = np.empty(n_theta)

    for i, th in enumerate(thetas):
        M = np.cos(th)*H + np.sin(th)*K
        ev = np.sort(np.linalg.eigvalsh(M))
        gaps[i] = ev[1] - ev[0]
    
    mingap = gaps.min()
    relgap = mingap/(np.abs(a) + 1e-9)
    theta_min = thetas[np.argmin(gaps)]

    info = {
        "a": a, "b": b,
        "min_gap": mingap,
        "rel_gap": relgap,
        "theta_min": theta_min,
        "flat_edge_confirmed": relgap < flat_tol,
    }
    return A, 3, info

