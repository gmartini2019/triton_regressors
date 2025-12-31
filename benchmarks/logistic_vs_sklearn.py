import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from triton_regressors.logistic.model import TritonLogisticRegression


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
y = (np.random.rand(B) > 0.5).astype(np.int32)

sk = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
sk.fit(X, y)

t_sklearn = time_fn(lambda: sk.predict_proba(X))

model = TritonLogisticRegression()
model.fit(X, y)
X_t = torch.tensor(X, device="cuda")
t_triton = time_fn(lambda: model.predict_proba(X_t))

print(f"sklearn inference: {t_sklearn:.2f} ms")
print(f"triton inference:  {t_triton:.2f} ms")
