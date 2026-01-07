import time
import numpy as np
import torch
from sklearn.linear_model import ElasticNet

from triton_regressors.elasticnet.model import TritonElasticNet


def time_cpu(fn, iters=30):
    fn()
    t0 = time.time()
    for _ in range(iters):
        fn()
    return (time.time() - t0) / iters * 1e3


def time_gpu(fn, iters=100):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


B, D = 8192, 512
alpha = 0.1
l1_ratio = 0.7

X = np.random.randn(B, D).astype(np.float32)
true_w = np.random.randn(D).astype(np.float32)
y = (X @ true_w + 0.1 * np.random.randn(B).astype(np.float32)).astype(np.float32)

sk = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=5000, tol=1e-6)
sk.fit(X, y)

t_sk_pred = time_cpu(lambda: sk.predict(X), iters=20)

model = TritonElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=5000, tol=1e-5)
model.fit(X, y)

X_t = torch.tensor(X, device="cuda")
t_triton_pred = time_gpu(lambda: model.predict(X_t), iters=200)

print(f"sklearn inference (CPU): {t_sk_pred:.2f} ms")
print(f"triton inference (GPU):  {t_triton_pred:.2f} ms")