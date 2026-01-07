import torch
import triton

from triton_regressors.linear.kernels import linear_regression_kernel
from .kernels import xtx_kernel, xty_kernel


class TritonRidgeRegression:
    """
    Ridge Regression (GPU):
      - closed_form: compute XtX and Xty (Triton), solve (XtX + alpha I) w = Xty via Cholesky
      - torch: optimize w with Torch (LBFGS) on centered data, avoids forming XtX explicitly

    Intercept handling:
      If fit_intercept=True:
        Xc = X - mean(X, dim=0)
        yc = y - mean(y)
        Solve for w on centered data
        b = y_mean - X_mean @ w
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


    def _as_cuda_float32_matrix(self, X):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (B, D). Got shape {tuple(X.shape)}")
        return X.contiguous()

    def _as_cuda_float32_vector(self, y):
        y = torch.as_tensor(y, device="cuda", dtype=torch.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D (B,). Got shape {tuple(y.shape)}")
        return y.contiguous()

    def _center_if_needed(self, X, y):
        """
        Returns:
          Xc, yc, X_mean, y_mean
        """
        B, D = X.shape
        if self.fit_intercept:
            X_mean = X.mean(dim=0)
            y_mean = y.mean()
            Xc = (X - X_mean).contiguous()
            yc = (y - y_mean).contiguous()
        else:
            X_mean = torch.zeros((D,), device="cuda", dtype=torch.float32)
            y_mean = torch.zeros((), device="cuda", dtype=torch.float32)
            Xc = X
            yc = y
        return Xc, yc, X_mean, y_mean

    def _recover_intercept(self, X_mean, y_mean, w):
        return y_mean - (X_mean @ w)

    def _fit_closed_form(self, X, y):
        Xc, yc, X_mean, y_mean = self._center_if_needed(X, y)
        B, D = Xc.shape

        XtX = torch.empty((D, D), device="cuda", dtype=torch.float32)
        grid_xtx = lambda META: (
            triton.cdiv(D, META["BM"]),
            triton.cdiv(D, META["BN"]),
        )
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

        b = self._recover_intercept(X_mean, y_mean, w)

        self.coef_ = w.detach()
        self.intercept_ = b.detach()
        self.n_features_in_ = D
        self._fitted = True
        return self

    def _fit_torch(self, X, y):
        Xc, yc, X_mean, y_mean = self._center_if_needed(X, y)
        B, D = Xc.shape

        w = torch.zeros((D,), device="cuda", dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [w],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=self.tol,
            tolerance_change=self.tol,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        alpha = self.alpha

        def closure():
            optimizer.zero_grad(set_to_none=True)
            pred = Xc @ w
            loss = (pred - yc).pow(2).mean()
            if alpha != 0.0:
                loss = loss + alpha * (w @ w)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            b = self._recover_intercept(X_mean, y_mean, w)

        self.coef_ = w.detach()
        self.intercept_ = b.detach()
        self.n_features_in_ = D
        self._fitted = True
        return self

    def fit(self, X, y):
        X = self._as_cuda_float32_matrix(X)
        y = self._as_cuda_float32_vector(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same B. Got {X.shape[0]} vs {y.shape[0]}")

        if self.solver == "closed_form":
            return self._fit_closed_form(X, y)
        elif self.solver == "torch":
            return self._fit_torch(X, y)
        else:
            raise ValueError(f"Unknown solver='{self.solver}'. Use 'closed_form' or 'torch'.")

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        X = self._as_cuda_float32_matrix(X)
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