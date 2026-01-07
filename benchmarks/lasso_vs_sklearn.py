import time
import numpy as np
import torch
from sklearn.linear_model import Lasso

from triton_regressors.lasso.model import TritonLassoRegression


def time_cpu(fn, iters=5):
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1e3


def time_gpu(fn, iters=20):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


D = 512
ALPHA = 0.1
BATCHES = [128, 512, 2048, 8192]

print("\nLASSO TRAINING (fit)")
print(
    f"{'B':>6} | "
    f"{'sklearn fit (ms)':>18} | "
    f"{'torch fista fit (ms)':>20}"
)
print("-" * 54)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    def sklearn_fit():
        Lasso(alpha=ALPHA, fit_intercept=True, max_iter=20_000).fit(X, y)

    t_sk = time_cpu(sklearn_fit, iters=3)

    def torch_fit():
        TritonLassoRegression(
            alpha=ALPHA,
            fit_intercept=True,
            method="fista",
            max_iter=1500,
            tol=1e-6,
            verbose=False,
        ).fit(X, y)

    t_torch = time_gpu(torch_fit, iters=3)

    print(f"{B:6d} | {t_sk:18.3f} | {t_torch:20.3f}")


print("\nLASSO INFERENCE (predict)")
print(
    f"{'B':>6} | "
    f"{'sklearn pred (ms)':>18} | "
    f"{'triton infer (ms)':>18}"
)
print("-" * 46)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    sk = Lasso(alpha=ALPHA, fit_intercept=True, max_iter=20_000).fit(X, y)

    def sklearn_pred():
        sk.predict(X)

    t_sk_pred = time_cpu(sklearn_pred, iters=50)

    X_t = torch.tensor(X, device="cuda")
    model = TritonLassoRegression(alpha=ALPHA, fit_intercept=True, method="fista", max_iter=1500)
    model.fit(X, y)

    for _ in range(10):
        model.predict(X_t)

    t_triton = time_gpu(lambda: model.predict(X_t), iters=200)

    print(f"{B:6d} | {t_sk_pred:18.3f} | {t_triton:18.3f}")