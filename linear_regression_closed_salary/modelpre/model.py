import numpy as np

def fit(X, y):
    m = X.shape[0]
    X_bias = np.hstack([np.ones((m, 1)), X])

    theta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    b = theta[0, 0]
    w = theta[1:]

    return w, b


def predict(X, w, b):
    """
    Dự đoán y_hat = Xw + b
    """
    return X @ w + b