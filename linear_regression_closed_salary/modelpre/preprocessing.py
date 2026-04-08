import pandas as pd
import os
from encoding.encoding import encode_categorical
from modelpre.scalestd import calculate_and_save_params, apply_standard_scale


def load_feature_names(path="data/feature_names.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_processed_data(input_csv, mode="train"):
    all_cols = load_feature_names()
    if not all_cols:
        raise ValueError("Lỗi: feature_names.txt trống hoặc không tìm thấy!")

    df = pd.read_csv(input_csv)

    if mode == "train":
        # Khi train: cần đủ tất cả các cột (bao gồm cả cột target cuối)
        df = df[all_cols].dropna()
        df = encode_categorical(df)

        # Tính và lưu mean/std trên toàn bộ tập train (cả X lẫn y)
        calculate_and_save_params(df, all_cols)
        df_scaled = apply_standard_scale(df, all_cols)

        if not os.path.exists("data"):
            os.makedirs("data")
        df_scaled.to_csv("data/data_scaled.csv", index=False)

        X = df_scaled.iloc[:, :-1].values
        y = df_scaled.iloc[:, -1:].values
        return X, y

    elif mode == "predict":
        # Khi predict: CHỈ lấy các cột feature (bỏ cột target cuối)
        features_only = all_cols[:-1]

        # Kiểm tra cột bị thiếu — cảnh báo rõ thay vì âm thầm gán 0
        # (gán 0 trước encode sẽ sai với cột categorical vì map sẽ trả về NaN)
        missing = [col for col in features_only if col not in df.columns]
        if missing:
            raise ValueError(
                f"Lỗi: dudoan.csv thiếu các cột sau: {missing}\n"
                "Hãy bổ sung đầy đủ trước khi dự đoán."
            )

        df = df[features_only]
        df = encode_categorical(df)

        # Scale dựa trên params đã lưu từ lúc train
        df_scaled = apply_standard_scale(df, features_only)

        # Trả về cả numpy array (để predict) lẫn DataFrame (để lưu file)
        return df_scaled.values, df_scaled