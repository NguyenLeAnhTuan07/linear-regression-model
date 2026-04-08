import pandas as pd
import os
import numpy as np
from modelpre.preprocessing import get_processed_data, load_feature_names
from modelpre.scalestd import rescale_y
from modelpre.model import predict


def main():
    # 1. Đường dẫn file đầu vào
    input_file = "predict/dudoan.csv"

    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}. Hãy tạo file và nhập dữ liệu cần dự đoán.")
        return

    # 2. SCALE: Encoding + Standard Scale dựa trên params đã lưu ở meta/
    try:
        X_test, df_scaled = get_processed_data(input_file, mode="predict")
    except Exception as e:
        print(f"Lỗi khi tiền xử lý dữ liệu: {e}")
        return

    # 3. Lưu file dudoan_scaled.csv vào thư mục predict/
    if not os.path.exists("predict"):
        os.makedirs("predict")
    scaled_output_path = "predict/dudoan_scaled.csv"
    df_scaled.to_csv(scaled_output_path, index=False)
    print(f"Đã lưu dữ liệu đã scale tại: {scaled_output_path}")

    # 4. LOAD MÔ HÌNH: Đọc trọng số w và b từ wb.csv
    wb_path = "meta/wb.csv"
    if not os.path.exists(wb_path):
        print("Lỗi: Không tìm thấy meta/wb.csv. Bạn cần chạy train.py trước để huấn luyện mô hình!")
        return

    df_wb = pd.read_csv(wb_path)
    w = df_wb.iloc[0, :-1].values.reshape(-1, 1)
    b = df_wb.iloc[0, -1]

    # 5. DỰ ĐOÁN: y_hat = Xw + b (kết quả ở không gian scale)
    y_hat_scaled = predict(X_test, w, b)

    # 6. RESCALE: Chuyển kết quả về đơn vị gốc
    try:
        all_cols = load_feature_names()
        target_name = all_cols[-1]
        y_real = rescale_y(y_hat_scaled, target_name)
    except Exception as e:
        print(f"Lỗi khi rescale kết quả: {e}")
        return

    # 7. XUẤT KẾT QUẢ
    for i, gia in enumerate(y_real.flatten()):
        print(f" Mẫu số {i+1}: {gia:,.2f}")


if __name__ == "__main__":
    main()