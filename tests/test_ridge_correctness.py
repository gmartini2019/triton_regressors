import numpy as np
import torch

from triton_regressors.ridge.model import TritonRidgeRegression


def test_ridge_closed_form_matches_torch_reference():
    torch.manual_seed(0)
    np.random.seed(0)

    B, D = 1024, 256
    alpha = 1.5

    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    X_t = torch.tensor(X, device="cuda")
    y_t = torch.tensor(y, device="cuda")

    X_mean = X_t.mean(dim=0)
    y_mean = y_t.mean()

    Xc = X_t - X_mean
    yc = y_t - y_mean

    XtX = Xc.T @ Xc
    XtX = 0.5 * (XtX + XtX.T)
    XtX += alpha * torch.eye(D, device="cuda")

    Xty = Xc.T @ yc

    L = torch.linalg.cholesky(XtX)
    w_ref = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)
    b_ref = y_mean - X_mean @ w_ref

    y_ref = (X_t @ w_ref + b_ref).cpu().numpy()


    model = TritonRidgeRegression(
        alpha=alpha,
        fit_intercept=True,
        solver="closed_form",
    )
    model.fit(X, y)

    y_triton = model.predict(X_t).cpu().numpy()

    assert np.allclose(
        y_ref,
        y_triton,
        rtol=1e-3,
        atol=1e-2,
    )