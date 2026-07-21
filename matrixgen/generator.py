import numpy as np

_RESULT_FIELDS = (
    "matrix",
    "case",
    "eigenvalues",
    "certificate",
    "certificate_threshold",
    "certificate_passed",
)

def _result(A, case, certificate, threshold):
    """Build the common public result schema used by every generator."""
    return {
        "matrix": A,
        "case": case,
        "eigenvalues": np.linalg.eigvals(A),
        "certificate": float(certificate),
        "certificate_threshold": float(threshold),
        "certificate_passed": bool(certificate <= threshold),
    }

def random_orthonormal_basis(n):
    A = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    d = d / np.abs(d)
    return Q*d

def random_hermitian(n, scale=1.0):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) * scale
    return (z + z.conj().T)/2

def random_eigenvalues(k):
    """Return k  random, distinct eigenvalues."""
    return np.random.randn(k) + 1j * np.random.randn(k)

def _support_gaps(H, K, n_theta):
    """Smallest gap between the two largest eigenvalues."""
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    gaps = np.empty(n_theta)
    for i, theta in enumerate(thetas):
        M = np.cos(theta) * H + np.sin(theta) * K
        eigenvalues = np.linalg.eigvalsh(M)
        gaps[i] = eigenvalues[-1] - eigenvalues[-2]

    index = np.argmin(gaps)
    return gaps[index], thetas[index]

def _relative_commutator_norm(A):
    commutator = A @ A.conj().T - A.conj().T @ A
    return np.linalg.norm(commutator, "fro") / (np.linalg.norm(A, "fro")**2)


# case 1: normal matrices
def gen_case1(distinct_eigvals=3):
    if not 1 <= distinct_eigvals <= 3:
        raise ValueError("distinct_eigvals must be between 1 and n")
    
    eigvals = random_eigenvalues(distinct_eigvals)
    Q = random_orthonormal_basis(3)
    A = Q @ np.diag(eigvals) @ Q.conj().T
    
    certificate = _relative_commutator_norm(A)
    return _result(A, 1, certificate, 1e-12)


# case 2: reducible matrices
def gen_case2(inside, scale=2.0):
    lam1 = np.random.randn() + 1j*np.random.randn()
    lam2 = np.random.randn() + 1j*np.random.randn()
    p = (np.random.randn() + 1j*np.random.randn()) * scale
    B = np.array([[lam1, p], [0, lam2]])

    center = (lam1 + lam2) / 2
    b = abs(p/2)
    c = abs(lam1 - lam2) / 2
    a = np.sqrt(b**2 + c**2)
    uhat = (lam2 - lam1) / (2*c) if c > 1e-6 else 1
    uper = 1j * uhat 

    r = np.random.randn(0, 0.9) if inside else np.random.randn(1, 2.5)
    t = np.random.uniform(0, 2*np.pi)

    lam0 = center + r * (a*np.cos(t)*uhat + b*np.sin(t)*uper)
    block = np.zeros((3, 3), dtype=complex)
    block[:2, :2] = B
    block[2, 2] = lam0
    U = random_orthonormal_basis(3)
    A = U @ block @ U.conj().T
    
    q = U[:, 2]
    residual = max(
        np.linalg.norm(A @ q - lam0 * q),
        np.linalg.norm(A.conj().T @ q - lam0.conjugate() * q),
    ) / max(np.linalg.norm(A, "fro"), np.finfo(float).eps)

    return _result(A, 2, residual, 1e-12)


# case 3: irreducible matrix, flat portion of the boundary
def gen_case3(scale=2.0, n_theta=720, flat_tol=2e-3):
    a, b = np.sort(np.random.rand(2))[::-1] * scale
    if a - b < 0.5:
        a = b + 1
    
    U = random_orthonormal_basis(3)
    H = U @ np.diag([a, a, b]) @ U.conj().T
    K = random_hermitian(3, scale)
    A = H + 1j * K

    min_gap, theta_min = _support_gaps(H, K, n_theta)
    typical_scale = np.abs(np.linalg.eigvalsh(H)).max() + 1e-9
    rel_gap = min_gap / typical_scale
    return _result(A, 3, rel_gap, flat_tol)


# case 4: irreducible matrices, smooth boundary
def gen_case4(scale=2.0, n_theta=720, flat_tol=1e-3, max_attempts=100):
    for _ in range(max_attempts):
        H = random_hermitian(3, scale)
        K = random_hermitian(3, scale)
        A = H + 1j * K

        min_gap, theta_min = _support_gaps(H, K, n_theta)
        typical_scale = np.abs(np.linalg.eigvalsh(H)).max() + 1e-9
        rel_gap = min_gap / typical_scale
        if rel_gap > flat_tol:
            return _result(A, 4, flat_tol/rel_gap, 1.0)
    raise RuntimeError("could not generate a smooth-boundary case within max_attempts")