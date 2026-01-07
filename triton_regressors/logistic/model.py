import torch
import torch.nn.functional as F

from triton_regressors.linear.kernels import linear_regression_kernel


class TritonLogisticRegression:
    """
    Logistic Regression (binary) with Torch training and Triton inference.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        max_iter: int = 200,
        tol: float = 1e-6,
    ):
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._fitted = False

    def fit(self, X, y):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        y = torch.as_tensor(y, device="cuda", dtype=torch.float32).contiguous()

        if y.ndim != 1:
            raise ValueError("y must be 1D binary labels")
        B, D = X.shape

        if self.fit_intercept:
            X_mean = X.mean(0)
            Xc = X - X_mean
        else:
            X_mean = torch.zeros(D, device="cuda")
            Xc = X

        w = torch.zeros(D, device="cuda", requires_grad=True)
        b = torch.zeros((), device="cuda", requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [w, b],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=self.tol,
            tolerance_change=self.tol,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits = Xc @ w + b
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            if self.fit_intercept:
                b = b - (X_mean @ w)

        self.coef_ = w.detach()
        self.intercept_ = b.detach()
        self.n_features_in_ = D
        self._fitted = True
        return self

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
        B, D = X.shape

        probs = torch.empty((B,), device="cuda", dtype=torch.float32)

        linear_regression_kernel[(B,)](
            X,
            self.coef_,
            self.intercept_,
            probs,
            B,
            D,
            X.stride(0),
            X.stride(1),
        )
        probs.sigmoid_()
        return probs

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).to(torch.int32)