import numpy as np
import torch
from sklearn.linear_model import ElasticNet

from triton_regressors.elasticnet.model import TritonElasticNet


def test_elasticnet_correctness_matches_sklearn():
    np.random.seed(0)
    torch.manual_seed(0)

    B, D = 1024, 128
    alpha = 0.1
    l1_ratio = 0.7

    X = np.random.randn(B, D).astype(np.float32)
    true_w = np.random.randn(D).astype(np.float32)
    y = (X @ true_w + 0.1 * np.random.randn(B).astype(np.float32)).astype(np.float32)

    sk = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=5000, tol=1e-6)
    sk.fit(X, y)
    y_sk = sk.predict(X).astype(np.float32)

    model = TritonElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=5000, tol=1e-5)
    model.fit(X, y)
    y_triton = model.predict(torch.tensor(X, device="cuda")).cpu().numpy()

    assert np.allclose(y_sk, y_triton, rtol=1e-2, atol=1e-2)
