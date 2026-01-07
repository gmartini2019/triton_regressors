import time
import numpy as np
import torch
from sklearn.linear_model import Ridge

from triton_regressors.ridge.model import TritonRidgeRegression


def time_cpu(fn, iters=10):
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1e3


def time_gpu(fn, iters=50):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


D = 512
ALPHA = 1.0
BATCHES = [128, 512, 2048, 8192]

print("\nTRAINING (fit)")
print(
    f"{'B':>6} | "
    f"{'sklearn (ms)':>14} | "
    f"{'closed_form (ms)':>18} | "
    f"{'torch_solver (ms)':>18}"
)
print("-" * 70)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    def sklearn_fit():
        Ridge(alpha=ALPHA, fit_intercept=True).fit(X, y)

    t_sklearn = time_cpu(sklearn_fit, iters=10)

    def triton_cf_fit():
        TritonRidgeRegression(
            alpha=ALPHA,
            fit_intercept=True,
            solver="closed_form",
        ).fit(X, y)

    t_cf = time_gpu(triton_cf_fit, iters=5)

    def triton_torch_fit():
        TritonRidgeRegression(
            alpha=ALPHA,
            fit_intercept=True,
            solver="torch",
        ).fit(X, y)

    t_torch = time_gpu(triton_torch_fit, iters=3)

    print(
        f"{B:6d} | "
        f"{t_sklearn:14.3f} | "
        f"{t_cf:18.3f} | "
        f"{t_torch:18.3f}"
    )


print("\nINFERENCE (predict)")
print(
    f"{'B':>6} | "
    f"{'sklearn (ms)':>14} | "
    f"{'triton infer (ms)':>18} | "
    f"{'torch mm (ms)':>14}"
)
print("-" * 70)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    sk = Ridge(alpha=ALPHA).fit(X, y)

    def sklearn_pred():
        sk.predict(X)

    t_sklearn = time_cpu(sklearn_pred, iters=50)

    X_t = torch.tensor(X, device="cuda")

    model = TritonRidgeRegression(
        alpha=ALPHA,
        fit_intercept=True,
        solver="closed_form",
    )
    model.fit(X, y)

    # warmup
    for _ in range(10):
        model.predict(X_t)

    t_triton = time_gpu(lambda: model.predict(X_t), iters=200)

    W = model.coef_
    b = model.intercept_

    for _ in range(10):
        X_t @ W + b

    t_torch_mm = time_gpu(lambda: (X_t @ W + b), iters=200)

    print(
        f"{B:6d} | "
        f"{t_sklearn:14.3f} | "
        f"{t_triton:18.3f} | "
        f"{t_torch_mm:14.3f}"
    )