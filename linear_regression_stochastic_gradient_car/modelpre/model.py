import numpy as np

def initialize_parameters(n):
    """
    Khởi tạo trọng số và bias
    n: số feature
    """
    w = np.zeros((n, 1))
    b = 0.0
    return w, b


def predict(X, w, b):
    """
    Dự đoán y_hat = Xw + b
    """
    return X @ w + b


def compute_loss(y, y_hat):
    """
    Hàm mất mát MSE
    """
    m = y.shape[0]
    loss = (1 / m) * np.sum((y_hat - y) ** 2)
    return loss


def compute_gradients(X, y, y_hat):
    """
    Tính gradient cho w và b
    """
    m = X.shape[0]

    dw = (2 / m) * X.T @ (y_hat - y)
    db = (2 / m) * np.sum(y_hat - y)

    return dw, db


def fit(X, y, lr=0.02, epochs=2000, patience=100):
    m, n = X.shape
    w, b = initialize_parameters(n)

    best_w = w.copy()
    best_b = b
    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):

        # Shuffle dữ liệu mỗi epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # ===== SGD: cập nhật từng mẫu =====
        for i in range(m):
            xi = X_shuffled[i:i+1]   # (1, n)
            yi = y_shuffled[i:i+1]   # (1, 1)

            y_hat = predict(xi, w, b)

            dw, db = compute_gradients(xi, yi, y_hat)

            w -= lr * dw
            b -= lr * db

        # ===== tính loss toàn bộ dataset =====
        y_hat_full = predict(X, w, b)
        loss = compute_loss(y, y_hat_full)

        if loss < best_loss:
            best_loss = loss
            best_w = w.copy()
            best_b = b
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_w, best_b


