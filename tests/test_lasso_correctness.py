import numpy as np
from sklearn.linear_model import Lasso

from triton_regressors.lasso.model import TritonLassoRegression
from triton_regressors.lasso.reference import predict_numpy


def test_lasso_correctness():
    np.random.seed(0)

    B, D = 1024, 128
    X = np.random.randn(B, D).astype(np.float32)
    y = np.random.randn(B).astype(np.float32)

    sk = Lasso(alpha=0.1, max_iter=10_000)
    sk.fit(X, y)

    y_ref = predict_numpy(X, sk.coef_, sk.intercept_)

    model = TritonLassoRegression(alpha=0.1)
    model.fit(X, y)

    y_triton = model.predict(X).cpu().numpy()

    assert np.allclose(y_ref, y_triton, atol=1e-4)
