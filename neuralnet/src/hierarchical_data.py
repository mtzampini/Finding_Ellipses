import numpy as np

def filter_node_A(y):
    y_node_A = (y == 1).astype(int)
    return y_node_A

def filter_node_B(X, y):
    not_normal = [2, 3, 4]
    mask = np.isin(y, not_normal)
    X_node_B = X[mask]
    y_node_B = y[mask]
    y_node_B = (y_node_B == 2).astype(int)
    return X_node_B, y_node_B

def filter_node_C(X, y):
    not_reducible_normal = [3, 4]
    mask = np.isin(y, not_reducible_normal)
    X_node_C = X[mask]
    y_node_C = y[mask]
    y_node_C = (y_node_C == 3).astype(int)
    return X_node_C, y_node_C

