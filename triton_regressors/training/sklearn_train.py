import numpy as np
from sklearn.linear_model import LinearRegression

def train_linear_regression(X: np.ndarray, y: np.ndarray):
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_, model.intercept_
