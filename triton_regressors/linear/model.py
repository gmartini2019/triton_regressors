import torch

from triton_regressors.core.base import BaseRegressor
from triton_regressors.training.closed_form import linear_closed_form
from triton_regressors.linear.kernels import linear_regression_kernel


class TritonLinearRegression(BaseRegressor):

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = bool(fit_intercept)
        self._fitted = False

    def fit(self, X, y):
        w, b = linear_closed_form(X, y, self.fit_intercept)

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