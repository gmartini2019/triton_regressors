import time
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from triton_regressors.linear.model import TritonLinearRegression


def time_cpu_fn(fn, iters=10):
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1e3


def time_gpu_fn(fn, iters=50):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


D = 512
BATCHES = [128, 512, 2048, 8192]

print("\nTRAINING ONLY (fit)")
print(f"{'B':>6} | {'sklearn fit (ms)':>18} | {'torch fit (ms)':>16}")
print("-" * 46)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    def sklearn_fit():
        model = LinearRegression()
        model.fit(X, y)

    t_sklearn_fit = time_cpu_fn(sklearn_fit, iters=10)

    def torch_fit():
        model = TritonLinearRegression(fit_intercept=True)
        model.fit(X, y)

    t_torch_fit = time_gpu_fn(torch_fit, iters=10)

    print(
        f"{B:6d} | "
        f"{t_sklearn_fit:18.4f} | "
        f"{t_torch_fit:16.4f}"
    )

print("\nINFERENCE ONLY (predict)")
print(
    f"{'B':>6} | "
    f"{'sklearn pred (ms)':>18} | "
    f"{'triton infer (ms)':>18} | "
    f"{'torch mm (ms)':>14}"
)
print("-" * 62)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    sk = LinearRegression().fit(X, y)

    def sklearn_pred():
        sk.predict(X)

    t_sklearn_pred = time_cpu_fn(sklearn_pred, iters=50)

    X_t = torch.tensor(X, device="cuda")
    model = TritonLinearRegression(fit_intercept=True)
    model.fit(X, y)

    for _ in range(10):
        model.predict(X_t)

    t_triton = time_gpu_fn(
        lambda: model.predict(X_t),
        iters=200,
    )

    W = model.coef_
    b = model.intercept_

    for _ in range(10):
        X_t @ W + b

    t_torch_mm = time_gpu_fn(
        lambda: (X_t @ W + b),
        iters=200,
    )

    print(
        f"{B:6d} | "
        f"{t_sklearn_pred:18.4f} | "
        f"{t_triton:18.4f} | "
        f"{t_torch_mm:14.4f}"
    )
