import pandas as pd
import numpy as np
import os
from modelpre.model import fit, predict
from modelpre.preprocessing import load_feature_names


def calculate_metrics(y_true, y_pred):
    """Tính toán bộ 3 chỉ số MSE, MAE, R2"""
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))

    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return mse, mae, r2


def run_k_fold_cv(X, y, k=5):
    """
    Thực hiện K-Fold Cross Validation.
    Trả về history chứa MSE, MAE, R2 (trên không gian đã scale) của từng fold.
    """
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    fold_sizes = np.full(k, m // k)
    fold_sizes[: m % k] += 1
    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[current : current + size])
        current += size

    history = {"MSE": [], "MAE": [], "R2": []}

    print(f"\n--- Đang thực hiện {k}-Fold Cross Validation ---")
    print(f"{'Fold':<10} | {'MSE':<14} | {'MAE':<14} | {'R2 Score':<12}")
    print("-" * 58)

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        w_f, b_f = fit(X_train, y_train)
        y_pred_f = predict(X_test, w_f, b_f)

        mse_f, mae_f, r2_f = calculate_metrics(y_test, y_pred_f)

        history["MSE"].append(mse_f)
        history["MAE"].append(mae_f)
        history["R2"].append(r2_f)

        print(f"Fold {i+1:<5} | {mse_f:<14.6f} | {mae_f:<14.6f} | {r2_f:<12.4f}")

    return history


def main():
    path = "data/data_scaled.csv"
    if not os.path.exists(path):
        print("Lỗi: Không tìm thấy data_scaled.csv. Hãy chạy train.py trước!")
        return

    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values  # y ở không gian đã scale

    # Lấy std_y để rescale MSE/MAE về đơn vị gốc
    all_cols = load_feature_names()
    target_col = all_cols[-1]
    params = pd.read_csv("meta/scale_params.csv", index_col=0)
    std_y = params.loc[target_col, "std"]

    K = 5
    cv_history = run_k_fold_cv(X, y, k=K)

    # ------------------------------------------------------------------ #
    # PHẦN 1: Đánh giá tổng quát — trung bình K-Fold                     #
    # MSE/MAE rescale về đơn vị gốc để có ý nghĩa thực tế                #
    # R² không đổi khi rescale nên giữ nguyên                            #
    # ------------------------------------------------------------------ #
    mse_scaled_mean = np.mean(cv_history["MSE"])
    mae_scaled_mean = np.mean(cv_history["MAE"])
    r2_mean = np.mean(cv_history["R2"])

    # MAE_real = MAE_scaled * std_y  (tuyến tính)
    # MSE_real = MSE_scaled * std_y² (bình phương)
    mae_real = mae_scaled_mean * std_y
    mse_real = mse_scaled_mean * (std_y ** 2)

    print("\n" + "=" * 58)
    print("ĐÁNH GIÁ TỔNG QUÁT (K-FOLD MEAN)")
    print("=" * 58)
    print(f"MSE (đơn vị gốc²) : {mse_real:.4f}")
    print(f"MAE (đơn vị gốc)  : {mae_real:.4f}")
    print(f"R2 Score          : {r2_mean:.4f}  ({r2_mean * 100:.2f}%)")
    print("-" * 58)


    # ------------------------------------------------------------------ #
    # PHẦN 2: CV Score — người dùng chọn metric Mi                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 58)
    print("CV SCORE")
    print("=" * 58)
    print("Bạn muốn tính CV Score dựa trên chỉ số nào (Mi)?")
    print("  1. MSE")
    print("  2. MAE")
    print("  3. R2 Score")
    choice = input("Nhập lựa chọn (1/2/3): ").strip()

    metric_map = {"1": "MSE", "2": "MAE", "3": "R2"}

    if choice not in metric_map:
        print("Lựa chọn không hợp lệ. Kết thúc chương trình.")
        return

    chosen = metric_map[choice]
    scores = np.array(cv_history[chosen])
    cv_mean = np.mean(scores)
    cv_std = np.std(scores, ddof=0)  # population std, nhất quán với công thức CV

    print("\n" + "-" * 58)
    print(f"KẾT QUẢ CV SCORE  (Mi = {chosen})")
    print(f"  CV_Score_mean  :  {cv_mean:.6f}")
    print(f"  CV_Score_std   :  {cv_std:.6f}")
    print("=" * 58)


if __name__ == "__main__":
    main()