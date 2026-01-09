import torch

def ridge_lbfgs_torch(
    X,
    y,
    alpha: float,
    fit_intercept: bool,
    max_iter: int,
    tol: float,
):
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

    w = torch.zeros(
        (D,),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [w],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

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
        b = y_mean - (X_mean @ w)

    return w.detach(), b.detach()

def lasso_ista_fista(
    X,
    y,
    alpha: float,
    fit_intercept: bool,
    method: str,
    max_iter: int,
    tol: float,
    lipschitz_iters: int,
    track_objective: bool,
):
    B, D = X.shape

    if fit_intercept:
        X_mean = X.mean(0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean
    else:
        X_mean = torch.zeros(D, device=X.device)
        y_mean = torch.zeros((), device=X.device)
        Xc, yc = X, y

    v = torch.randn(D, device=X.device)
    v /= v.norm() + 1e-12
    for _ in range(lipschitz_iters):
        v = Xc.T @ (Xc @ v)
        v /= v.norm() + 1e-12
    L = (Xc @ v).dot(Xc @ v) / B
    lr = 1.0 / L

    lam = alpha * lr

    w = torch.zeros(D, device=X.device)
    w_prev = w.clone()
    t = 1.0

    objective = []

    def soft_threshold(z, l):
        return torch.sign(z) * torch.clamp(torch.abs(z) - l, min=0.0)

    for k in range(max_iter):
        z = w

        r = Xc @ z - yc
        grad = (Xc.T @ r) / B

        w_new = soft_threshold(z - lr * grad, lam)

        if method == "fista":
            t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
            w = w_new + ((t - 1) / t_new) * (w_new - w_prev)
            w_prev = w_new
            t = t_new
            w_eval = w_new
        else:
            w = w_new
            w_eval = w

        if (w_eval - w_prev).norm() / (w_prev.norm() + 1e-12) < tol:
            break

        if track_objective:
            with torch.no_grad():
                mse = 0.5 * (r @ r) / B
                l1 = alpha * torch.abs(w_eval).sum()
                objective.append(float((mse + l1).item()))

    b = y_mean - X_mean @ w_eval
    return w_eval.detach(), b.detach(), objective, k + 1

def elasticnet_ista_fista(
    X,
    y,
    alpha: float,
    l1_ratio: float,
    fit_intercept: bool,
    method: str,
    max_iter: int,
    tol: float,
    lipschitz_iters: int,
    track_objective: bool,
):
    B, D = X.shape

    if fit_intercept:
        X_mean = X.mean(0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean
    else:
        X_mean = torch.zeros(D, device=X.device)
        y_mean = torch.zeros((), device=X.device)
        Xc, yc = X, y

    v = torch.randn(D, device=X.device)
    v /= v.norm() + 1e-12
    for _ in range(lipschitz_iters):
        v = Xc.T @ (Xc @ v)
        v /= v.norm() + 1e-12
    L = (Xc @ v).dot(Xc @ v) / B
    L += alpha * (1.0 - l1_ratio) 
    lr = 1.0 / L

    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)

    def soft_threshold(z, lam):
        return torch.sign(z) * torch.clamp(torch.abs(z) - lam, min=0.0)

    w = torch.zeros(D, device=X.device)
    w_prev = w.clone()
    t = 1.0

    objective = []

    for k in range(max_iter):
        z = w

        r = Xc @ z - yc
        grad = (Xc.T @ r) / B + l2 * z

        w_new = soft_threshold(z - lr * grad, lr * l1)

        if method == "fista":
            t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
            w = w_new + ((t - 1) / t_new) * (w_new - w_prev)
            w_prev = w_new
            t = t_new
            w_eval = w_new
        else:
            w = w_new
            w_eval = w

        if (w_eval - w_prev).norm() / (w_prev.norm() + 1e-12) < tol:
            break

        if track_objective:
            with torch.no_grad():
                mse = 0.5 * (r @ r) / B
                l1_term = l1 * torch.abs(w_eval).sum()
                l2_term = 0.5 * l2 * (w_eval @ w_eval)
                objective.append(float((mse + l1_term + l2_term).item()))

    b = y_mean - X_mean @ w_eval
    return w_eval.detach(), b.detach(), objective, k + 1