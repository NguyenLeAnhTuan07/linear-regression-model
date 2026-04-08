import pandas as pd
import os
from modelpre.preprocessing import get_processed_data, load_feature_names
from modelpre.model import fit


def main():
    # 1. Preprocessing: encoding + standard scale → lưu data_scaled.csv và scale_params.csv
    #    Trả về X, y đều đã được scale về không gian chuẩn hóa
    X_train, y_train = get_processed_data("data/data.csv", mode="train")

    print(f"Huấn luyện với {X_train.shape[0]} mẫu, {X_train.shape[1]} features")

    # 2. Huấn luyện: X và y đều đã scale → gradient descent hội tụ ổn định
    w, b = fit(X_train, y_train)

    # 3. Lưu trọng số w và b vào meta/wb.csv để dùng cho predict.py
    feat_names = load_feature_names()[:-1]  # Bỏ cột target, chỉ lấy tên feature
    wb_df = pd.DataFrame(
        [list(w.flatten()) + [b]],
        columns=feat_names + ["bias"]
    )

    if not os.path.exists("meta"):
        os.makedirs("meta")
    wb_df.to_csv("meta/wb.csv", index=False)

    print("Hệ số đã được lưu tại meta/wb.csv")


if __name__ == "__main__":
    main()