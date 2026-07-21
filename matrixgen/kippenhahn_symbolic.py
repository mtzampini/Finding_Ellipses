import sympy as sp

h11, h22, h33 = sp.symbols('h11 h22 h33', real=True)
h12r, h12i = sp.symbols('h12r h12i', real=True)
h13r, h13i = sp.symbols('h13r h13i', real=True)
h23r, h23i = sp.symbols('h23r h23i', real=True)

H = sp.Matrix([
    [h11,              h12r + sp.I*h12i, h13r + sp.I*h13i],
    [h12r - sp.I*h12i, h22,              h23r + sp.I*h23i],
    [h13r - sp.I*h13i, h23r - sp.I*h23i, h33]
])

k11, k22, k33 = sp.symbols('k11 k22 k33', real=True)
k12r, k12i = sp.symbols('k12r k12i', real=True)
k13r, k13i = sp.symbols('k13r k13i', real=True)
k23r, k23i = sp.symbols('k23r k23i', real=True)

K = sp.Matrix([
    [k11,              k12r + sp.I*k12i, k13r + sp.I*k13i],
    [k12r - sp.I*k12i, k22,              k23r + sp.I*k23i],
    [k13r - sp.I*k13i, k23r - sp.I*k23i, k33]
])

u, v, w = sp.symbols('u v w')
L = sp.expand((u*H + v*K + w*sp.eye(3)).det())
poly = sp.Poly(L, u, v, w)
coefficient_extractor = sp.lambdify(
    (
        h11,h22,h33,h12r,h12i,h13r,h13i,h23r,h23i,
        k11,k22,k33,k12r,k12i,k13r,k13i,k23r,k23i
    ),
    poly.coeffs(),
    "numpy"
)
