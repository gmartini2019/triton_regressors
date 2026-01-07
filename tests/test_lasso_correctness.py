import numpy as np
import torch
from sklearn.linear_model import Lasso

from triton_regressors.lasso.model import TritonLassoRegression


def test_lasso_torch_solver_matches_sklearn_predictions():
    np.random.seed(0)
    torch.manual_seed(0)

    B, D = 2048, 128
    alpha = 0.1

    X = np.random.randn(B, D).astype(np.float32)
    y = (np.random.randn(B).astype(np.float32) * 0.5)

    sk = Lasso(alpha=alpha, fit_intercept=True, max_iter=20_000)
    sk.fit(X, y)
    y_ref = sk.predict(X)

    model = TritonLassoRegression(
        alpha=alpha,
        fit_intercept=True,
        method="fista",
        max_iter=3000,
        tol=1e-6,
        verbose=False,
    )
    model.fit(X, y)

    X_t = torch.tensor(X, device="cuda")
    y_hat = model.predict(X_t).cpu().numpy()

    assert np.allclose(y_ref, y_hat, rtol=1e-2, atol=1e-2)