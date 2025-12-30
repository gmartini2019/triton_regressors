import numpy as np
from sklearn.linear_model import LinearRegression
import torch

def train_linear_regression(X: np.ndarray, y: np.ndarray):
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_, model.intercept_

def train_linear_regression_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr=1e-2,
    epochs=1000,
):
    X = torch.tensor(X, device="cuda", dtype=torch.float32)
    y = torch.tensor(y, device="cuda", dtype=torch.float32)

    B, D = X.shape

    w = torch.zeros(D, device="cuda", requires_grad=False)
    b = torch.zeros((), device="cuda", requires_grad=False)

    for _ in range(epochs):
        y_pred = X @ w + b
        err = y_pred - y

        grad_w = (2.0 / B) * (X.T @ err)
        grad_b = (2.0 / B) * err.sum()

        w -= lr * grad_w
        b -= lr * grad_b

    return w.detach().cpu().numpy(), float(b.detach().cpu())