from __future__ import annotations

import abc
import torch


class BaseRegressor(abc.ABC):
    @abc.abstractmethod
    def fit(self, X, y):
        """
        Fit the model.

        After this method returns, the following attributes MUST exist:
          -> self.coef_          (torch.Tensor, CUDA)
          -> self.intercept_     (torch.Tensor, CUDA)
          -> self.n_features_in_ (int)

        Returns:
            self
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError

    coef_: torch.Tensor
    intercept_: torch.Tensor
    n_features_in_: int

    objective_: list[float] | None = None
    n_iter_: int | None = None
    converged_: bool | None = None

    def _check_fitted(self):
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            raise RuntimeError("Model has not been fitted yet.")

    def _ensure_cuda_params(self):
        if not self.coef_.is_cuda or not self.intercept_.is_cuda:
            raise RuntimeError(
                "Model parameters must live on CUDA. "
                "This library enforces GPU-resident inference."
            )
