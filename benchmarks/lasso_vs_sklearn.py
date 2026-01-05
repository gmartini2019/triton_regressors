import time
import numpy as np
import torch
from sklearn.linear_model import Lasso

from triton_regressors.lasso.model import TritonLassoRegression


def time_fn(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


B, D = 8192, 512
X = np.random.randn(B, D).astype(np.float32)
y = np.random.randn(B).astype(np.float32)

sk = Lasso(alpha=0.1, max_iter=10_000)
sk.fit(X, y)

t_sklearn = time_fn(lambda: sk.predict(X))

model = TritonLassoRegression(alpha=0.1)
model.fit(X, y)
X_t = torch.tensor(X, device="cuda")
t_triton = time_fn(lambda: model.predict(X_t))

print(f"sklearn inference: {t_sklearn:.2f} ms")
print(f"triton inference:  {t_triton:.2f} ms")
