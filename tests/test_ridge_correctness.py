import numpy as np
import torch

from triton_regressors.ridge.model import TritonRidgeRegression


def test_ridge_correctness_matches_torch_reference():
    torch.manual_seed(0)
    np.random.seed(0)

    B, D = 1024, 256
    alpha = 1.5

    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    X_t = torch.tensor(X, device="cuda", dtype=torch.float32)
    y_t = torch.tensor(y, device="cuda", dtype=torch.float32)

    X_mean = X_t.mean(dim=0)
    y_mean = y_t.mean()
    Xc = X_t - X_mean
    yc = y_t - y_mean

    XtX_ref = Xc.T @ Xc
    XtX_ref = 0.5 * (XtX_ref + XtX_ref.T)
    XtX_ref += alpha * torch.eye(D, device="cuda", dtype=torch.float32)

    Xty_ref = Xc.T @ yc

    L = torch.linalg.cholesky(XtX_ref)
    w_ref = torch.cholesky_solve(Xty_ref.unsqueeze(1), L).squeeze(1)
    b_ref = y_mean - X_mean @ w_ref

    y_ref = (X_t @ w_ref + b_ref).cpu().numpy()


    model = TritonRidgeRegression(alpha=alpha, fit_intercept=True)
    model.fit(X, y)


    y_triton = model.predict(X_t).cpu().numpy()

    XtX_ref_noreg = Xc.T @ Xc
    XtX_ref_noreg = 0.5 * (XtX_ref_noreg + XtX_ref_noreg.T)
    XtX_ref_reg = XtX_ref_noreg + alpha * torch.eye(D, device="cuda", dtype=torch.float32)

    assert np.allclose(y_ref, y_triton, rtol=1e-3, atol=1e-2)
