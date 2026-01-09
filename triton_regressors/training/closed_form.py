import torch
import triton

from triton_regressors.ridge.kernels import xtx_kernel, xty_kernel


def linear_closed_form(X, y, fit_intercept: bool):
    X = torch.as_tensor(X, device="cuda", dtype=torch.float64).contiguous()
    y = torch.as_tensor(y, device="cuda", dtype=torch.float64).contiguous().view(-1)

    if fit_intercept:
        X_mean = X.mean(0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean
    else:
        X_mean = torch.zeros(X.shape[1], device="cuda", dtype=X.dtype)
        y_mean = torch.zeros((), device="cuda", dtype=y.dtype)
        Xc = X
        yc = y

    w = torch.linalg.lstsq(Xc, yc).solution
    b = y_mean - X_mean @ w

    return w.to(torch.float32), b.to(torch.float32)

def ridge_closed_form(X, y, alpha: float, fit_intercept: bool):
    """
    Closed-form ridge regression:
      (X^T X + alpha I) w = X^T y
    """
    X = torch.as_tensor(X, device="cuda", dtype=torch.float32).contiguous()
    y = torch.as_tensor(y, device="cuda", dtype=torch.float32).contiguous().view(-1)

    B, D = X.shape

    if fit_intercept:
        X_mean = X.mean(0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean
    else:
        X_mean = torch.zeros(D, device="cuda")
        y_mean = torch.zeros((), device="cuda")
        Xc = X
        yc = y

    XtX = Xc.T @ Xc
    XtX = 0.5 * (XtX + XtX.T)
    XtX += alpha * torch.eye(D, device="cuda", dtype=X.dtype)

    Xty = Xc.T @ yc

    L = torch.linalg.cholesky(XtX)
    w = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)

    b = y_mean - X_mean @ w

    return w, b

def ridge_closed_form_triton(X, y, alpha: float, fit_intercept: bool):
    """
    Ridge regression via explicit XtX / Xty using Triton kernels.
    Solves:
        (XtX + alpha I) w = Xty
    """
    B, D = X.shape

    if fit_intercept:
        X_mean = X.mean(dim=0)
        y_mean = y.mean()
        Xc = (X - X_mean).contiguous()
        yc = (y - y_mean).contiguous()
    else:
        X_mean = torch.zeros((D,), device=X.device, dtype=X.dtype)
        y_mean = torch.zeros((), device=X.device, dtype=X.dtype)
        Xc, yc = X, y

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

    if alpha != 0.0:
        XtX = XtX + alpha * torch.eye(D, device="cuda", dtype=torch.float32)

    L = torch.linalg.cholesky(XtX)
    w = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)

    b = y_mean - (X_mean @ w)
    return w.detach(), b.detach()