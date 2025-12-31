import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

from .kernels import logistic_predict_kernel


class TritonLogisticRegression:
    def __init__(self):
        self._fitted = False

    def fit(self, X, y):
        sk = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
        )
        sk.fit(X, y)

        self.coef_ = torch.tensor(
            sk.coef_[0], device="cuda", dtype=torch.float32
        )
        self.intercept_ = torch.tensor(
            sk.intercept_[0], device="cuda", dtype=torch.float32
        )
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self

    def predict_proba(self, X):
        assert self._fitted

        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        B, D = X.shape

        Y = torch.empty((B,), device="cuda", dtype=torch.float32)

        logistic_predict_kernel[(B,)](
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

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).to(torch.int32)
