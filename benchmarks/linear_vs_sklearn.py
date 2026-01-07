import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from triton_regressors.logistic.model import TritonLogisticRegression


def time_cpu(fn, iters=5):
    fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1e3


def time_gpu(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3

BATCHES = [128, 512, 2048, 8192]
D = 512

print("\nLOGISTIC REGRESSION — TRAINING (fit)")
print(
    f"{'B':>6} | "
    f"{'sklearn CPU (ms)':>18} | "
    f"{'torch GPU (ms)':>18}"
)
print("-" * 52)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = (np.random.rand(B) > 0.5).astype(np.float32)

    def sklearn_fit():
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=200,
        ).fit(X, y)

    t_sklearn = time_cpu(sklearn_fit, iters=3)

    def torch_fit():
        TritonLogisticRegression(
            fit_intercept=True,
            max_iter=200,
        ).fit(X, y)

    t_torch = time_gpu(torch_fit, iters=3)

    print(
        f"{B:6d} | "
        f"{t_sklearn:18.3f} | "
        f"{t_torch:18.3f}"
    )


print("\nLOGISTIC REGRESSION — INFERENCE (predict_proba)")
print(
    f"{'B':>6} | "
    f"{'sklearn CPU (ms)':>18} | "
    f"{'triton GPU (ms)':>18} | "
    f"{'torch mm+sigmoid (ms)':>22}"
)
print("-" * 74)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = (np.random.rand(B) > 0.5).astype(np.float32)

    sk = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=200,
    ).fit(X, y)

    def sklearn_pred():
        sk.predict_proba(X)

    t_sklearn = time_cpu(sklearn_pred, iters=50)

    model = TritonLogisticRegression(
        fit_intercept=True,
        max_iter=200,
    )
    model.fit(X, y)

    X_t = torch.tensor(X, device="cuda")

    for _ in range(10):
        model.predict_proba(X_t)

    def triton_pred():
        model.predict_proba(X_t)

    t_triton = time_gpu(triton_pred, iters=200)

    W = model.coef_
    b = model.intercept_

    def torch_mm():
        torch.sigmoid(X_t @ W + b)

    for _ in range(10):
        torch_mm()

    t_torch_mm = time_gpu(torch_mm, iters=200)

    print(
        f"{B:6d} | "
        f"{t_sklearn:18.3f} | "
        f"{t_triton:18.3f} | "
        f"{t_torch_mm:22.3f}"
    )