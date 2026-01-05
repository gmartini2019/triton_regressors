import torch
from triton_regressors.linear.kernels import linear_regression_kernel
from triton_regressors.utils.torch_train import center_if_needed, recover_intercept

class TritonLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = bool(fit_intercept)
        self._fitted = False

    def fit(self, X, y):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float64).contiguous()
        y = torch.as_tensor(y, device="cuda", dtype=torch.float64).contiguous().view(-1)

        Xc, yc, X_mean, y_mean = center_if_needed(X, y, self.fit_intercept)

        w = torch.linalg.lstsq(Xc, yc).solution

        b = recover_intercept(X_mean, y_mean, w)

        self.coef_ = w.to(torch.float32)
        self.intercept_ = b.to(torch.float32)
        self.n_features_in_ = X.shape[1]
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        B, D = X.shape
        Y = torch.empty((B,), device="cuda", dtype=torch.float32)

        linear_regression_kernel[(B,)](
            X, self.coef_, self.intercept_, Y,
            B, D, X.stride(0), X.stride(1),
        )
        return Y
