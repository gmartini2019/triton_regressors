import numpy as np
import torch
import matplotlib.pyplot as plt

from triton_regressors.elasticnet.model import TritonElasticNet


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

B, D = 2048, 256
alpha = 0.1
l1_ratio = 0.5

Z = np.random.randn(B, 50).astype(np.float32)
A = np.random.randn(50, D).astype(np.float32)
X = Z @ A + 0.05 * np.random.randn(B, D).astype(np.float32)

true_w = np.zeros(D, dtype=np.float32)
true_w[:20] = np.random.randn(20).astype(np.float32)
y = (X @ true_w + 0.1 * np.random.randn(B).astype(np.float32)).astype(np.float32)

ista = TritonElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    fit_intercept=True,
    method="ista",
    max_iter=500,
    tol=0.0,
    track_objective=True,
)
ista.fit(X, y)
obj_ista = np.array(ista.objective_)

fista = TritonElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    fit_intercept=True,
    method="fista",
    max_iter=500,
    tol=0.0,
    track_objective=True,
)
fista.fit(X, y)
obj_fista = np.array(fista.objective_)

plt.figure(figsize=(7, 5))
plt.plot(obj_ista, label="ISTA")
plt.plot(obj_fista, label="FISTA")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Objective value")
plt.title("ElastiNet convergence (ISTA vs FISTA)")
plt.legend()
plt.tight_layout()
plt.savefig("elastinet_convergence.png", dpi=150)
print("Done and saved to elastinet_convergence.png")
