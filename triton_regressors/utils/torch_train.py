import torch

def center_if_needed(X, y, fit_intercept: bool):
    if not fit_intercept:
        X_mean = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
        y_mean = torch.zeros((), device=y.device, dtype=y.dtype)
        return X, y, X_mean, y_mean
    X_mean = X.mean(dim=0)
    y_mean = y.mean()
    return (X - X_mean), (y - y_mean), X_mean, y_mean

def recover_intercept(X_mean, y_mean, w):
    return y_mean - X_mean @ w
