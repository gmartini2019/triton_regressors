import torch

from triton_regressors.linear.kernels import linear_regression_kernel


def soft_threshold(z: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)


class TritonElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        assert 0.0 <= self.l1_ratio <= 1.0
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._fitted = False

    def fit(self, X, y):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        y = torch.as_tensor(y, device="cuda", dtype=torch.float32).contiguous()

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]

        n, d = X.shape

        if self.fit_intercept:
            X_mean = X.mean(dim=0)
            y_mean = y.mean()
            Xc = (X - X_mean).contiguous()
            yc = (y - y_mean).contiguous()
        else:
            X_mean = torch.zeros((d,), device="cuda", dtype=torch.float32)
            y_mean = torch.zeros((), device="cuda", dtype=torch.float32)
            Xc = X
            yc = y

        col_norm2 = (Xc * Xc).sum(dim=0)  

        l1 = self.alpha * self.l1_ratio * n
        l2 = self.alpha * (1.0 - self.l1_ratio) * n

        w = torch.zeros((d,), device="cuda", dtype=torch.float32)
        r = yc.clone()  

        denom = col_norm2 + l2
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))

        for _ in range(self.max_iter):
            max_update = 0.0

            for j in range(d):
                Xj = Xc[:, j]
                w_old = w[j]

                rho = (Xj @ r) + col_norm2[j] * w_old

                w_new = soft_threshold(rho, l1) / denom[j]

                if w_new != w_old:
                    r = r + Xj * (w_old - w_new)
                    w[j] = w_new
                    max_update = max(max_update, torch.abs(w_new - w_old).item())

            if max_update < self.tol:
                break

        b = y_mean - (X_mean @ w)

        self.coef_ = w
        self.intercept_ = b
        self.n_features_in_ = d
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted, "Model must be fitted first"
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        assert X.ndim == 2 and X.shape[1] == self.n_features_in_

        B, D = X.shape
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
