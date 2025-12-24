import numpy as np

def predict_numpy(X: np.ndarray, coef: np.ndarray, intercept: float):
    """
    Reference NumPy implementation of linear regression inference.
    """
    return X @ coef + intercept
