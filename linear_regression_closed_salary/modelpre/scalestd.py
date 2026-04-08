import pandas as pd
import numpy as np
import os

def calculate_and_save_params(df, columns):
    """Tính mean/std và lưu vào meta."""
    if not os.path.exists("meta"): os.makedirs("meta")
    data = df.values
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.where(std == 0, 1, std) # Tránh chia cho 0
    pd.DataFrame({'mean': mean, 'std': std}, index=columns).to_csv("meta/scale_params.csv")

def apply_standard_scale(df, columns):
    """Biến đổi (x - mean) / std."""
    params = pd.read_csv("meta/scale_params.csv", index_col=0)
    mean = params.loc[columns, 'mean'].values
    std = params.loc[columns, 'std'].values
    return pd.DataFrame((df.values - mean) / std, columns=columns)

def rescale_y(y_scaled, target_col_name):
    """Biến số nhỏ thành số tiền thật."""
    params = pd.read_csv("meta/scale_params.csv", index_col=0)
    mean_y = params.loc[target_col_name, 'mean']
    std_y = params.loc[target_col_name, 'std']
    return y_scaled * std_y + mean_y