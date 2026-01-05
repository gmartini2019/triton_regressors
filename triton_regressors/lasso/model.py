import torch
from sklearn.linear_model import Lasso

from .kernels import lasso_predict_kernel


class TritonLassoRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._fitted = False

    def fit(self, X, y):
        sk = Lasso(
            alpha=self.alpha,
            fit_intercept=True,
            max_iter=10_000,
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
        assert self._fitted

        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        B, D = X.shape

        Y = torch.empty((B,), device="cuda", dtype=torch.float32)

        lasso_predict_kernel[(B,)](
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
