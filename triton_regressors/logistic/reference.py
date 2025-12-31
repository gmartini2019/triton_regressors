import numpy as np

def predict_proba_numpy(X, w, b):
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))
