import torch
from triton_regressors.linear.kernels import linear_regression_kernel


def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


def estimate_lipschitz(X: torch.Tensor, iters: int = 20, eps: float = 1e-12) -> float:
    B, D = X.shape
    v = torch.randn(D, device=X.device, dtype=X.dtype)
    v = v / (v.norm() + eps)

    for _ in range(iters):
        Xv = X @ v
        v = X.T @ Xv
        v = v / (v.norm() + eps)

    Xv = X @ v
    sigma2 = (Xv @ Xv).item()
    return max(sigma2 / B, eps)


class TritonLassoRegression:
    """
    Lasso Regression via proximal gradient (ISTA / FISTA)

    Objective:
      (1/(2B)) ||Xw - y||^2 + alpha ||w||_1
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        method: str = "fista",     # or "ista"
        max_iter: int = 2000,
        tol: float = 1e-6,
        lipschitz_iters: int = 20,
        track_objective: bool = False,
    ):
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.method = method
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.lipschitz_iters = int(lipschitz_iters)
        self.track_objective = bool(track_objective)
        self.objective_ = []
        self._fitted = False

    def _center_if_needed(self, X, y):
        B, D = X.shape
        if self.fit_intercept:
            X_mean = X.mean(0)
            y_mean = y.mean()
            Xc = X - X_mean
            yc = y - y_mean
        else:
            X_mean = torch.zeros(D, device=X.device)
            y_mean = torch.zeros((), device=X.device)
            Xc, yc = X, y
        return Xc, yc, X_mean, y_mean

    def fit(self, X, y):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        y = torch.as_tensor(y, device="cuda", dtype=torch.float32).contiguous()
        if y.ndim == 2:
            y = y.squeeze(1)

        B, D = X.shape

        Xc, yc, X_mean, y_mean = self._center_if_needed(X, y)

        L = estimate_lipschitz(Xc, self.lipschitz_iters)
        lr = 1.0 / L
        lam = self.alpha

        w = torch.zeros(D, device="cuda")
        w_eval = w.clone()
        t = 1.0

        for k in range(self.max_iter):
            w_old = w_eval.clone()

            r = Xc @ w_eval - yc
            grad = (Xc.T @ r) / B

            w_new = soft_threshold(w_eval - lr * grad, lr * lam)

            if self.method == "fista":
                t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
                w_eval = w_new + ((t - 1) / t_new) * (w_new - w)
                w = w_new
                t = t_new
            else:
                w_eval = w_new
                w = w_new

            if (w_eval - w_old).norm() / (w_old.norm() + 1e-12) < self.tol:
                break

            if self.track_objective:
                with torch.no_grad():
                    mse = 0.5 * (r @ r) / B
                    l1 = self.alpha * torch.abs(w_eval).sum()
                    self.objective_.append(float((mse + l1).item()))

        b = y_mean - X_mean @ w_eval

        self.coef_ = w_eval.detach()
        self.intercept_ = b.detach()
        self.n_features_in_ = D
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
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