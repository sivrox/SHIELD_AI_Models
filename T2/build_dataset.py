"""
STEP 1 (UPDATED): Build Combined Dataset (MIT-BIH + UCI + Synthetic)
====================================================================

Changes vs earlier version:
- Do NOT hardcode has_disease=True for all MIT-BIH windows.
- Infer has_disease from signal + vitals:
    * high HR / low HRV / high BP / high BMI ⇒ more likely diseased
- Include activity (0=rest, 1=walk, 2=run).
- Risk score uses activity:
    * high HR at rest = risky
    * high HR during running = less risky
- Basic cleaning: drop impossible values and null rows.

Output:
  combined_data.csv
"""

import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks

# ============================================================================
# USER SETTINGS (EDIT ONLY THESE)
# ============================================================================

MITBIH_DIR = r"mit_bih_data"  # change path
UCI_FOLDER = r"uci_data"
TARGET_MIN_ROWS = 10000  # total rows after augmentation
UCI_TARGET_ROWS = 1000

# ============================================================================
# STAGE 1: LOAD MIT-BIH AS WINDOWS
# ============================================================================

def list_mitbih_records(mitbih_dir):
    basenames = set()
    for fname in os.listdir(mitbih_dir):
        if fname.endswith(".dat"):
            basenames.add(os.path.splitext(fname)[0])
    return sorted(list(basenames))


def compute_hr_and_hrv(segment, fs):
    distance = int(0.25 * fs)
    peaks, _ = find_peaks(segment, distance=distance)
    if len(peaks) < 2:
        return np.nan, np.nan
    rr = np.diff(peaks) / fs
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    hr = 60.0 / mean_rr
    hrv_ms = std_rr * 1000.0
    return hr, hrv_ms


def load_mitbih_windows(mitbih_dir, window_sec=30.0, max_windows_per_record=80):
    records = list_mitbih_records(mitbih_dir)
    rows = []
    print(f"Found {len(records)} MIT-BIH records: {records}")

    for rec in records:
        rec_path = os.path.join(mitbih_dir, rec)
        print(f"Reading record {rec_path} ...")
        try:
            record = wfdb.rdrecord(rec_path)
        except Exception as e:
            print(f"  ⚠️ Could not read {rec}: {e}")
            continue

        fs = record.fs
        sig = record.p_signal[:, 0]
        total_samples = len(sig)
        win_samples = int(window_sec * fs)
        n_windows = min(total_samples // win_samples, max_windows_per_record)

        for w in range(n_windows):
            start = w * win_samples
            end = start + win_samples
            segment = sig[start:end]
            hr, hrv = compute_hr_and_hrv(segment, fs)
            if np.isnan(hr) or np.isnan(hrv):
                continue
            rows.append({
                "source": "MITBIH",
                "record": rec,
                "hr": hr,
                "hrv": hrv
            })
        print(f"  -> extracted {n_windows} windows from {rec}")

    df = pd.DataFrame(rows)
    print(f"MIT-BIH windows loaded: {len(df)} rows")
    return df

# ============================================================================
# STAGE 2: CREATE UCI-STYLE TABULAR BASE
# ============================================================================

def load_uci_base(uci_folder):
    """
    Robust loader for UCI Heart Disease datasets.
    Supports:
      - space or comma separated files
      - missing values marked as '?'
      - all 4 standard UCI files

    Output columns:
      age, sex, hr, hrv, bp_systolic, bp_diastolic, spo2, has_disease
    """

    data_files = [
        "processed.cleveland.data",
        "processed.hungarian.data",
        "processed.switzerland.data",
        "processed.va.data",
    ]

    all_rows = []

    for fname in data_files:
        path = os.path.join(uci_folder, fname)

        if not os.path.exists(path):
            print(f"⚠️ Missing UCI file: {fname}")
            continue

        print(f"Loading UCI file: {fname}")

        with open(path, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # normalize delimiters
                fields = line.replace(",", " ").split()
                if len(fields) != 14:
                    continue

                # convert to floats, replace ? with NaN
                values = []
                for x in fields:
                    if x == "?":
                        values.append(np.nan)
                    else:
                        values.append(float(x))

                (
                    age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal, target
                ) = values

                # REQUIRED fields (skip only if these are missing)
                if any(np.isnan(x) for x in [age, sex, trestbps, thalach, target]):
                    continue

                # basic sanity filters
                if not (18 <= age <= 95):
                    continue
                if not (60 <= thalach <= 220):
                    continue
                if not (80 <= trestbps <= 220):
                    continue

                # ---- derived vitals ----
                hr = thalach

                # HRV proxy (age-adjusted)
                hrv = max(
                    5.0,
                    45.0 - (age - 30) * 0.35 + np.random.normal(0, 4.0)
                )

                bp_sys = trestbps
                bp_dia = bp_sys * 0.6 + np.random.normal(0, 6.0)

                has_disease = int(target) != 0

                all_rows.append({
                    "source": "UCI_real",
                    "record": f"{fname}_{idx}",
                    "age": age,
                    "sex": int(sex),
                    "hr": hr,
                    "hrv": hrv,
                    "bp_systolic": bp_sys,
                    "bp_diastolic": bp_dia,
                    "spo2": 98.0,
                    "has_disease": has_disease,
                })

    df = pd.DataFrame(all_rows)

    print(f"✅ Loaded {len(df)} valid UCI rows")

    if len(df) == 0:
        print("⚠️ WARNING: No usable UCI rows found after cleaning.")

    return df


def expand_uci_with_synthetic(real_df, target_rows):
    """
    If we have few real UCI rows (e.g. 30), expand up to target_rows (e.g. 100)
    by sampling from real rows and adding small noise.
    """
    n_real = len(real_df)
    if n_real >= target_rows:
        return real_df.iloc[:target_rows].reset_index(drop=True)

    needed = target_rows - n_real
    print(f"Expanding UCI: need {needed} synthetic rows to reach {target_rows}")

    copies = []
    copies_per_row = int(np.ceil(needed / n_real))

    for _ in range(copies_per_row):
        noisy = real_df.copy()
        # add small noise to vitals
        for col, scale in [
            ("hr", 2.0),
            ("hrv", 3.0),
            ("bp_systolic", 5.0),
            ("bp_diastolic", 3.0),
        ]:
            noisy[col] = noisy[col] + np.random.normal(0, scale, size=len(noisy))

        # clamp to reasonable ranges
        noisy["hr"] = np.clip(noisy["hr"], 40, 200)
        noisy["bp_systolic"] = np.clip(noisy["bp_systolic"], 80, 220)
        noisy["bp_diastolic"] = np.clip(noisy["bp_diastolic"], 40, 140)
        noisy["hrv"] = np.clip(noisy["hrv"], 1, 200)

        noisy["source"] = "UCI_synth"
        copies.append(noisy)

    df_expanded = pd.concat([real_df] + copies, ignore_index=True).iloc[:target_rows]
    df_expanded = df_expanded.reset_index(drop=True)
    print(f"Final UCI (real + synthetic style-match): {len(df_expanded)}")
    return df_expanded

# ============================================================================
# STAGE 3: DEMOGRAPHICS, HAS_DISEASE, ACTIVITY, SLEEP, BMI
# ============================================================================

def infer_has_disease_from_signals(df):
    """
    Infer has_disease using simple rules instead of hardcoding True.
    Rules (add points; disease if score >= 2):
      +1 if hr > 100 or hr < 50
      +1 if hrv < age_adjusted_hrv * 0.6
      +1 if bp_systolic > 140 or bp_diastolic > 90
      +1 if bmi >= 30 (this is applied later, so here we only do HR/HRV/BP)
    For MIT-BIH we create a provisional flag here, refined after BMI is added.
    """
    df = df.copy()
    provisional = []
    for _, row in df.iterrows():
        age = row.get("age", 60)
        hr = row["hr"]
        hrv = row["hrv"]
        sys = row.get("bp_systolic", 130)
        dia = row.get("bp_diastolic", 80)

        score = 0
        if hr > 100 or hr < 50:
            score += 1
        age_expected_hrv = 50 - (age - 30) * 0.3
        if hrv < age_expected_hrv * 0.6:
            score += 1
        if sys > 140 or dia > 90:
            score += 1

        provisional.append(score >= 2)
    df["has_disease"] = provisional
    return df


def add_demographics_to_mitbih(mit_df):
    """
    MIT-BIH base has hr, hrv. Add:
      age, sex, bp_systolic, bp_diastolic, spo2, provisional has_disease.
    """
    mit_df = mit_df.copy()
    ages = np.random.randint(40, 80, size=len(mit_df))
    sexes = np.random.choice([0, 1], size=len(mit_df))
    bp_sys = 120 + (ages - 40) * 0.5 + np.random.normal(0, 8, size=len(mit_df))
    bp_dia = bp_sys * 0.6 + np.random.normal(0, 5, size=len(mit_df))
    spo2 = np.random.uniform(95, 99, size=len(mit_df))

    mit_df["age"] = ages
    mit_df["sex"] = sexes
    mit_df["bp_systolic"] = bp_sys
    mit_df["bp_diastolic"] = bp_dia
    mit_df["spo2"] = spo2

    mit_df = infer_has_disease_from_signals(mit_df)
    return mit_df


def add_sleep_and_bmi(df):
    df = df.copy()
    bmi_list = []
    sd_list = []
    sq_list = []
    rem_list = []
    deep_list = []

    for _, row in df.iterrows():
        age = row["age"]
        has_dis = bool(row["has_disease"])

        if has_dis:
            bmi = np.random.uniform(27, 36)
        else:
            bmi = np.random.uniform(22, 30)

        if has_dis:
            sleep_dur = np.random.uniform(5.0, 7.0)
            sleep_q = np.random.uniform(50, 75)
        else:
            sleep_dur = np.random.uniform(7.0, 8.5)
            sleep_q = np.random.uniform(70, 90)

        age_penalty = max(0, age - 40) * 0.4
        sleep_q = np.clip(sleep_q - age_penalty, 30, 95)

        rem = 15 + (sleep_q - 60) * 0.15
        deep = 12 + (sleep_q - 60) * 0.10
        rem = np.clip(rem, 5, 35)
        deep = np.clip(deep, 3, 25)

        bmi_list.append(bmi)
        sd_list.append(sleep_dur)
        sq_list.append(sleep_q)
        rem_list.append(rem)
        deep_list.append(deep)

    df["bmi"] = bmi_list
    df["sleep_duration"] = sd_list
    df["sleep_quality"] = sq_list
    df["rem_percentage"] = rem_list
    df["deep_percentage"] = deep_list

    return df


def refine_has_disease_with_bmi(df):
    """
    After BMI is added, bump has_disease when BMI is very high.
    """
    df = df.copy()
    flags = []
    for _, row in df.iterrows():
        score = 0
        if row["has_disease"]:
            score += 1
        if row["bmi"] >= 30:
            score += 1
        if row["bp_systolic"] > 150 or row["bp_diastolic"] > 95:
            score += 1
        flags.append(score >= 2)
    df["has_disease"] = flags
    return df


def add_activity(df):
    """
    Add activity level:
      0 = rest, 1 = walking, 2 = running
    Probability depends on HR and disease status.
    """
    df = df.copy()
    activities = []
    for _, row in df.iterrows():
        hr = row["hr"]
        has_dis = bool(row["has_disease"])

        if hr < 80:
            probs = [0.8, 0.18, 0.02]
        elif hr < 110:
            probs = [0.3, 0.6, 0.1]
        else:
            probs = [0.1, 0.4, 0.5]

        if has_dis:
            probs = [probs[0] + 0.1, probs[1] + 0.05, max(probs[2] - 0.15, 0)]
            s = sum(probs)
            probs = [p / s for p in probs]

        activity = np.random.choice([0, 1, 2], p=probs)
        activities.append(activity)

    df["activity"] = activities
    return df

# ============================================================================
# STAGE 4: RISK SCORE USING ACTIVITY
# ============================================================================

def calculate_risk_score_row(row):
    age = row["age"]
    hr = row["hr"]
    hrv = row["hrv"]
    spo2 = row["spo2"]
    sys = row["bp_systolic"]
    dia = row["bp_diastolic"]
    bmi = row["bmi"]
    sleep_dur = row["sleep_duration"]
    sleep_q = row["sleep_quality"]
    has_dis = bool(row["has_disease"])
    activity = int(row["activity"])

    risk = 0.1

    if age > 30:
        risk += (age - 30) * 0.005

    # HR + activity interaction
    if activity == 0:  # rest
        if hr > 90 or hr < 55:
            risk += 0.08
        if hr > 110 or hr < 45:
            risk += 0.07
    elif activity == 1:  # walking
        if hr > 120:
            risk += 0.05
    else:  # running
        if hr > 160:
            risk += 0.05

    # HRV (lower worse)
    age_expected_hrv = 50 - (age - 30) * 0.3
    if hrv < age_expected_hrv * 0.7:
        risk += 0.08

    # BP
    if sys > 140 or dia > 90:
        risk += 0.10
    if sys > 180 or dia > 120:
        risk += 0.15
    if sys < 90 or dia < 60:
        risk += 0.08

    # SpO2
    if spo2 < 95:
        risk += 0.05
    if spo2 < 90:
        risk += 0.15

    # BMI
    if bmi >= 25:
        risk += 0.05
    if bmi >= 30:
        risk += 0.05

    # Sleep
    if sleep_dur < 6:
        risk += 0.06
    if sleep_q < 60:
        risk += 0.05

    # Disease flag
    if has_dis:
        risk += 0.15

    return float(np.clip(risk, 0.0, 1.0))


def add_risk_score(df):
    df = df.copy()
    df["risk_score"] = df.apply(calculate_risk_score_row, axis=1)
    return df

# ============================================================================
# STAGE 5: AUGMENT, CLEAN, SAVE
# ============================================================================

def augment_to_target_rows(df, target_rows):
    if len(df) >= target_rows:
        return df.reset_index(drop=True)

    needed = target_rows - len(df)
    print(f"Need {needed} extra rows; augmenting data...")
    copies_per_row = int(np.ceil(needed / len(df)))
    parts = [df]

    for _ in range(copies_per_row):
        noisy = df.copy()
        for col, scale in [
            ("hr", 2.0),
            ("hrv", 3.0),
            ("bp_systolic", 5.0),
            ("bp_diastolic", 3.0),
            ("spo2", 0.3),
            ("bmi", 0.7),
            ("sleep_duration", 0.3),
            ("sleep_quality", 3.0),
        ]:
            noisy[col] = noisy[col] + np.random.normal(0, scale, size=len(noisy))

        noisy["spo2"] = np.clip(noisy["spo2"], 85, 100)
        noisy["sleep_duration"] = np.clip(noisy["sleep_duration"], 3, 12)
        noisy["sleep_quality"] = np.clip(noisy["sleep_quality"], 30, 95)
        noisy["risk_score"] = np.clip(
            noisy["risk_score"] + np.random.normal(0, 0.02, size=len(noisy)),
            0, 1
        )
        noisy["source"] = noisy["source"] + "_aug"
        parts.append(noisy)

    df_aug = pd.concat(parts, ignore_index=True).iloc[:target_rows]
    df_aug = df_aug.reset_index(drop=True)
    print(f"After augmentation: {len(df_aug)} rows")
    return df_aug


def clean_impossible_rows(df):
    """
    Drop rows with impossible values or NaNs.
    """
    df = df.dropna().copy()
    df = df[
        (df["age"].between(18, 95)) &
        (df["hr"].between(30, 220)) &
        (df["hrv"].between(0, 300)) &
        (df["bp_systolic"].between(50, 250)) &
        (df["bp_diastolic"].between(20, 150)) &
        (df["spo2"].between(80, 100)) &
        (df["bmi"].between(15, 50)) &
        (df["sleep_duration"].between(0, 12)) &
        (df["sleep_quality"].between(0, 100))
    ]
    return df.reset_index(drop=True)


def save_outputs(df, prefix="combined_data"):
    csv_path = f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("STEP 1 (UPDATED): BUILD COMBINED TABULAR DATASET".center(80))
    print("=" * 80)

    mit_base = load_mitbih_windows(MITBIH_DIR)
    mit_full = add_demographics_to_mitbih(mit_base)

        # === UCI REAL + EXPANDED ===
    uci_real = load_uci_base(UCI_FOLDER)  # now reads all 4 .data files
    uci_base = expand_uci_with_synthetic(uci_real, UCI_TARGET_ROWS)

    uci_full = add_sleep_and_bmi(uci_base)
    uci_full = refine_has_disease_with_bmi(uci_full)
    uci_full = add_activity(uci_full)  # add after BMI but before risk_score

    combined = pd.concat([mit_full, uci_full], ignore_index=True)


    # add sleep + bmi to both
    mit_full = add_sleep_and_bmi(mit_full)
    uci_full = add_sleep_and_bmi(uci_base)

    # refine has_disease with bmi
    mit_full = refine_has_disease_with_bmi(mit_full)
    uci_full = refine_has_disease_with_bmi(uci_full)

    combined = pd.concat([mit_full, uci_full], ignore_index=True)

    # add activity
    combined = add_activity(combined)

    # risk score
    combined = add_risk_score(combined)

    # clean impossible rows
    combined = clean_impossible_rows(combined)

    # augment
    combined = augment_to_target_rows(combined, TARGET_MIN_ROWS)

    print("\nSummary statistics:")
    print(combined.describe())

    save_outputs(combined, prefix="combined_data")

    print("\nDone. Output: combined_data.csv")


if __name__ == "__main__":
    main()
