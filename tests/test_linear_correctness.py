import numpy as np
import torch

from triton_regressors.triton_regressors.linear.reference import predict_numpy
from triton_regressors.triton_regressors.linear.model import TritonLinearRegression


def test_linear_regression_correctness():
    B, D = 512, 256

    X = np.random.randn(B, D).astype(np.float32)
    W = np.random.randn(D).astype(np.float32)
    b = np.float32(0.3)

    y = predict_numpy(X, W, b)

    model = TritonLinearRegression()
    model.fit(X, y)

    X_t = torch.tensor(X, device="cuda")
    y_triton = model.predict(X_t).cpu().numpy()

    assert np.allclose(y, y_triton, atol=1e-5)
