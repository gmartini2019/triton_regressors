import torch
from .kernels import linear_regression_kernel


class TritonLinearRegression:
    """
    sklearn-like Linear Regression (GPU inference via Triton)
    """

    def __init__(self, fit_intercept=True, positive=False):
        self.fit_intercept = fit_intercept
        self.positive = positive
        self._fitted = False

    def fit(self, X, y):
        """
        CPU training using sklearn, GPU inference via Triton
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression

        sk = LinearRegression(
            fit_intercept=self.fit_intercept,
            positive=self.positive,
        )
        sk.fit(X, y)

        self.coef_ = torch.tensor(
            sk.coef_, device="cuda", dtype=torch.float32
        )
        self.intercept_ = torch.tensor(
            sk.intercept_, device="cuda", dtype=torch.float32
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True

        return self

    def predict(self, X):
        assert self._fitted, "Model must be fitted first"
        assert X.ndim == 2
        assert X.shape[1] == self.n_features_in_

        if not torch.is_tensor(X):
            X = torch.tensor(X, device="cuda", dtype=torch.float32)
        else:
            X = X.to(device="cuda", dtype=torch.float32)

        B, D = X.shape
        Y = torch.empty((B,), device="cuda", dtype=torch.float32)

        grid = (B,)

        linear_regression_kernel[grid](
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
