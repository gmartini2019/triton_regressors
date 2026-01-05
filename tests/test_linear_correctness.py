import numpy as np
import torch

from triton_regressors.linear.reference import predict_numpy
from triton_regressors.linear.model import TritonLinearRegression


def test_linear_regression_training_and_inference():
    np.random.seed(0)
    torch.manual_seed(0)

    B, D = 512, 256

    X = np.random.randn(B, D).astype(np.float32)
    true_w = np.random.randn(D).astype(np.float32)
    true_b = np.float32(0.3)

    y = predict_numpy(X, true_w, true_b)

    model = TritonLinearRegression(fit_intercept=True)
    model.fit(X, y)

    X_t = torch.tensor(X, device="cuda")
    y_triton = model.predict(X_t).cpu().numpy()

    assert np.allclose(y, y_triton, atol=1e-4)
