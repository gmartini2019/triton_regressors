import time
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from triton_regressors.linear.model import TritonLinearRegression


def time_fn(fn, iters=50):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


D = 512
BATCHES = [1, 8, 32, 128, 512, 1024, 4096, 16384]

print(f"{'B':>6} | {'sklearn(ms)':>12} | {'triton(ms)':>12} | {'torch(mm)(ms)':>14}")
print("-" * 52)

for B in BATCHES:
    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    sk = LinearRegression().fit(X, y)

    X_t = torch.tensor(X, device="cuda")
    W_t = torch.tensor(sk.coef_, device="cuda")
    b_t = torch.tensor(sk.intercept_, device="cuda")

    X_t = torch.tensor(X, device="cuda")

    triton_model = TritonLinearRegression()
    triton_model.fit(X, y)        
    triton_model.predict(X_t)     


    triton_model.predict(X_t)
    X_t @ W_t + b_t

    t_sklearn = time_fn(lambda: sk.predict(X)) * 1e3
    t_triton = time_fn(lambda: triton_model.predict(X_t)) * 1e3
    t_torch = time_fn(lambda: (X_t @ W_t + b_t)) * 1e3

    print(f"{B:6d} | {t_sklearn:12.4f} | {t_triton:12.4f} | {t_torch:14.4f}")
