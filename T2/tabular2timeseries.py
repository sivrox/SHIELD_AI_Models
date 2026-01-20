"""
STEP 2 (UPDATED): Tabular → Time-Series Windows + Preprocessing
===============================================================

Input:
  combined_data.csv   (from step 1)

Output:
  windows_ts.npy        # (N_windows, T, F)  dynamic features
  windows_static.npy    # (N_windows, D)     static features (scaled)
  windows_y.npy         # (N_windows,)
  scaler_static.pkl     # fitted StandardScaler for static features

This script:
  - Drops null rows
  - Generates K windows per row
  - Scales static features with StandardScaler
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# USER SETTINGS
CSV_PATH = "combined_data.csv"
WINDOW_LENGTH = 60
WINDOWS_PER_ROW = 3

# --------- which columns go where ---------
DYNAMIC_COLS = ["hr", "hrv", "spo2", "bp_systolic", "bp_diastolic", "activity"]
STATIC_COLS = [
    "age", "sex", "bmi",
    "sleep_duration", "sleep_quality",
    "rem_percentage", "deep_percentage",
]


def generate_time_series_from_row(row, T):
    base_hr = row["hr"]
    base_hrv = row["hrv"]
    base_spo2 = row["spo2"]
    base_sys = row["bp_systolic"]
    base_dia = row["bp_diastolic"]
    activity = row["activity"]  # integer 0,1,2

    t = np.linspace(0, 1, T)

    hr_trend = 1.0 + 0.03 * np.sin(2 * np.pi * t)
    hr_noise = np.random.normal(0, 2.0, size=T)
    hr_series = base_hr * hr_trend + hr_noise

    hrv_trend = 1.0 + 0.05 * np.sin(2 * np.pi * t + np.pi / 4)
    hrv_noise = np.random.normal(0, 1.5, size=T)
    hrv_series = base_hrv * hrv_trend + hrv_noise
    hrv_series = np.clip(hrv_series, 1, None)

    spo2_noise = np.random.normal(0, 0.2, size=T)
    spo2_series = base_spo2 + spo2_noise
    spo2_series = np.clip(spo2_series, 85, 100)

    sys_trend = 1.0 + 0.02 * np.sin(2 * np.pi * t + np.pi / 3)
    sys_noise = np.random.normal(0, 3.0, size=T)
    sys_series = base_sys * sys_trend + sys_noise

    dia_trend = 1.0 + 0.02 * np.sin(2 * np.pi * t + np.pi / 2)
    dia_noise = np.random.normal(0, 2.0, size=T)
    dia_series = base_dia * dia_trend + dia_noise

    activity_series = np.full(T, activity)

    ts_window = np.stack(
        [hr_series, hrv_series, spo2_series, sys_series, dia_series, activity_series],
        axis=1
    )
    return ts_window.astype(np.float32)


def build_static_vector_from_row(row):
    return row[STATIC_COLS].values.astype(np.float32)


def main():
    print("=" * 80)
    print("STEP 2 (UPDATED): TABULAR → WINDOWS + PREPROCESS".center(80))
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["source", "record"], errors="ignore")
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Drop rows with any NA in required columns
    needed_cols = list(set(DYNAMIC_COLS + STATIC_COLS + ["risk_score"]))
    df = df.dropna(subset=needed_cols).reset_index(drop=True)
    print(f"After dropping NA rows: {len(df)}")

    # Basic range cleaning (again, just to be safe)
    df = df[
        (df["hr"].between(30, 220)) &
        (df["hrv"].between(0, 300)) &
        (df["bp_systolic"].between(50, 250)) &
        (df["bp_diastolic"].between(20, 150)) &
        (df["spo2"].between(80, 100)) &
        (df["activity"].isin([0, 1, 2]))
    ].reset_index(drop=True)
    print(f"After range cleaning: {len(df)} rows")

    N_rows = len(df)
    T = WINDOW_LENGTH
    K = WINDOWS_PER_ROW
    F = len(DYNAMIC_COLS)
    D = len(STATIC_COLS)
    total_windows = N_rows * K

    print(f"Each row → {K} windows of length {T}, dynamic F={F}, static D={D}")
    print(f"Total windows to generate: {total_windows}")

    X_ts = np.zeros((total_windows, T, F), dtype=np.float32)
    X_static = np.zeros((total_windows, D), dtype=np.float32)
    y = np.zeros((total_windows,), dtype=np.float32)

    idx = 0
    for _, row in df.iterrows():
        for _ in range(K):
            X_ts[idx] = generate_time_series_from_row(row, T)
            X_static[idx] = build_static_vector_from_row(row)
            y[idx] = np.float32(row["risk_score"])
            idx += 1

    print(f"Actually generated {idx} windows")

    # Scale static features
    scaler = StandardScaler()
    X_static_scaled = scaler.fit_transform(X_static)

    # Save NPZ and scaler
    np.save("windows_ts.npy", X_ts)
    np.save("windows_static.npy", X_static_scaled.astype(np.float32))
    np.save("windows_y.npy", y)

    with open("scaler_static.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nSaved:")
    print(f"  windows_ts.npy      shape = {X_ts.shape}")
    print(f"  windows_static.npy  shape = {X_static_scaled.shape}")
    print(f"  windows_y.npy       shape = {y.shape}")
    print("  scaler_static.pkl   (StandardScaler for static features)")
    print("\nDataset is now cleaned, windowed, and scaled — ready for model training.")

    # After all cleaning steps are done
    cleaned_tabular_path = "combined_data_cleaned_for_windows.csv"
    df.to_csv(cleaned_tabular_path, index=False)
    print(f"Saved cleaned tabular data to: {cleaned_tabular_path}")

if __name__ == "__main__":
    main()
