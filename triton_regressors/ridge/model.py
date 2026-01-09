import torch

from triton_regressors.core.base import BaseRegressor
from triton_regressors.training.closed_form import ridge_closed_form_triton
from triton_regressors.training.iterative import ridge_lbfgs_torch
from triton_regressors.linear.kernels import linear_regression_kernel


class TritonRidgeRegression(BaseRegressor):
    """
    Ridge Regression (GPU).

    Training backends:
      - solver="closed_form": Triton XtX / Xty + Cholesky
      - solver="torch": Torch LBFGS on centered data

    Inference:
      - Always Triton kernel (X @ w + b)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "closed_form",
        max_iter: int = 200,
        tol: float = 1e-8,
    ):
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.solver = str(solver)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._fitted = False

    def fit(self, X, y):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        y = torch.as_tensor(y, device="cuda", dtype=torch.float32).contiguous()

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same B. Got {X.shape[0]} vs {y.shape[0]}"
            )

        if self.solver == "closed_form":
            w, b = ridge_closed_form_triton(
                X, y, alpha=self.alpha, fit_intercept=self.fit_intercept
            )
        elif self.solver == "torch":
            w, b = ridge_lbfgs_torch(
                X,
                y,
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            raise ValueError(
                f"Unknown solver='{self.solver}'. Use 'closed_form' or 'torch'."
            )

        self.coef_ = w
        self.intercept_ = b
        self.n_features_in_ = w.numel()
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