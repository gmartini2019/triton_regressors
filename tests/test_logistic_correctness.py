import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from triton_regressors.logistic.model import TritonLogisticRegression
from triton_regressors.logistic.reference import predict_proba_numpy


def test_logistic_correctness():
    np.random.seed(0)

    B, D = 1024, 128
    X = np.random.randn(B, D).astype(np.float32)
    y = (np.random.rand(B) > 0.5).astype(np.int32)

    sk = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    sk.fit(X, y)

    y_ref = predict_proba_numpy(X, sk.coef_[0], sk.intercept_[0])

    model = TritonLogisticRegression()
    model.fit(X, y)

    y_triton = model.predict_proba(X).cpu().numpy()

    assert np.allclose(y_ref, y_triton, atol=2e-3)