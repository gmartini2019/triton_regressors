import torch
import triton

from triton_regressors.linear.kernels import linear_regression_kernel
from .kernels import xtx_kernel, xty_kernel, diag_add_kernel


class TritonRidgeRegression:
    """
    Ridge Regression:
      Solve (Xc^T Xc + alpha I) w = Xc^T yc
      b = y_mean - X_mean @ w
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self._fitted = False

    def fit(self, X, y):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32)
        y = torch.as_tensor(y, device="cuda", dtype=torch.float32)
        if y.ndim == 2:
            y = y.squeeze(1)

        X = X.contiguous()
        y = y.contiguous()

        B, D = X.shape

        # -------------------------
        # Centering
        # -------------------------
        if self.fit_intercept:
            X_mean = X.mean(dim=0)
            y_mean = y.mean()
            Xc = (X - X_mean).contiguous()
            yc = (y - y_mean).contiguous()
        else:
            X_mean = torch.zeros(D, device="cuda", dtype=torch.float32)
            y_mean = torch.zeros((), device="cuda", dtype=torch.float32)
            Xc = X
            yc = y

        XtX = torch.empty((D, D), device="cuda", dtype=torch.float32)

        grid_xtx = lambda META: (triton.cdiv(D, META["BM"]), triton.cdiv(D, META["BN"]))
        xtx_kernel[grid_xtx](
            Xc,
            XtX,
            B,
            D,
            Xc.stride(0),
            Xc.stride(1),
            XtX.stride(0),
            XtX.stride(1),
        )

        XtX = 0.5 * (XtX + XtX.T)
        Xty = torch.empty((D,), device="cuda", dtype=torch.float32)

        grid_xty = lambda META: (triton.cdiv(D, META["BLOCK_D"]),)
        xty_kernel[grid_xty](
            Xc,
            yc,
            Xty,
            B,
            D,
            Xc.stride(0),
            Xc.stride(1),
        )

        if self.alpha != 0.0:
            XtX = XtX + self.alpha * torch.eye(D, device="cuda", dtype=torch.float32)

        L = torch.linalg.cholesky(XtX)
        w = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)

        b = y_mean - (X_mean @ w)

        self.coef_ = w
        self.intercept_ = b
        self.n_features_in_ = D
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted, "Model must be fitted first"
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()

        B, D = X.shape
        assert D == self.n_features_in_

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
