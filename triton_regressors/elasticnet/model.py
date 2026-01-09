import torch

from triton_regressors.core.base import BaseRegressor
from triton_regressors.training.iterative import elasticnet_ista_fista
from triton_regressors.linear.kernels import linear_regression_kernel


class TritonElasticNet(BaseRegressor):
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        method: str = "fista",
        max_iter: int = 2000,
        tol: float = 1e-6,
        lipschitz_iters: int = 20,
        track_objective: bool = False,
    ):
        assert 0.0 <= l1_ratio <= 1.0

        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.fit_intercept = bool(fit_intercept)
        self.method = method
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.lipschitz_iters = int(lipschitz_iters)
        self.track_objective = bool(track_objective)
        self._fitted = False

    def fit(self, X, y):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        y = torch.as_tensor(y, device="cuda", dtype=torch.float32).contiguous()

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)

        w, b, objective, n_iter = elasticnet_ista_fista(
            X,
            y,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            method=self.method,
            max_iter=self.max_iter,
            tol=self.tol,
            lipschitz_iters=self.lipschitz_iters,
            track_objective=self.track_objective,
        )

        self.coef_ = w
        self.intercept_ = b
        self.n_features_in_ = w.numel()
        self.n_iter_ = n_iter
        self.objective_ = objective if self.track_objective else None
        self.converged_ = n_iter < self.max_iter
        self._fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        self._ensure_cuda_params()

        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        B, D = X.shape

        if D != self.n_features_in_:
            raise ValueError(f"Expected D={self.n_features_in_}, got D={D}")

        Y = torch.empty((B,), device="cuda", dtype=torch.float32)

        linear_regression_kernel[(B,)](
            X,
            self.coef_,
            self.intercept_,
            Y,
            B,
            D,
            X.stride(0),
            X.stride(1),
        )
        return Y