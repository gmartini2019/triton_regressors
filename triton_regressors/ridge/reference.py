import numpy as np

def predict_numpy(X: np.ndarray, coef: np.ndarray, intercept: float):
    return X @ coef + intercept
