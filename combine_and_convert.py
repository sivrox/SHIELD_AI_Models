import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("STEP 1: LOAD BOTH DATASETS")
print("=" * 60)

# Load real dataset
df_real = pd.read_csv('final_vitals.csv')
print(f"✓ Real dataset loaded: {df_real.shape[0]} rows")

# Load synthetic dataset
df_synthetic = pd.read_csv('synthetic_vitals.csv')
print(f"✓ Synthetic dataset loaded: {df_synthetic.shape[0]} rows")

print("\n" + "=" * 60)
print("STEP 2: COMBINE DATASETS (80% synthetic, 20% real)")
print("=" * 60)

# Sample 80% from synthetic, all from real
synthetic_sample = df_synthetic.sample(n=int(0.8 * len(df_synthetic)), random_state=42)
df_combined = pd.concat([df_real, synthetic_sample], ignore_index=True)

print(f"✓ Combined dataset: {df_combined.shape[0]} rows")
print(f"  - Real: {len(df_real)} rows")
print(f"  - Synthetic (80%): {len(synthetic_sample)} rows")

# Shuffle combined dataset
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✓ Shuffled and ready")

print("\n" + "=" * 60)
print("STEP 3: INFER ACTIVITY")
print("=" * 60)

def infer_activity(row):
    """
    Infer activity level based on HR and HRV patterns
    0 = resting, 1 = walking, 2 = running
    """
    hr = row['hr']
    hrv = row['hrv']
    activity_current = row['activity']
    
    if activity_current != 0:
        return activity_current
    
    if hr > 110 and hrv < 30:
        return 2
    elif hr > 85 and hrv < 50:
        return 1
    else:
        return 0

df_combined['activity'] = df_combined.apply(infer_activity, axis=1)
print(f"✓ Activity inferred")
print(f"  Distribution: {df_combined['activity'].value_counts().to_dict()}")

print("\n" + "=" * 60)
print("STEP 4: SAVE COMBINED DATASET (before normalization)")
print("=" * 60)

# Save the combined dataset for inspection
df_combined.to_csv('combined_dataset.csv', index=False)
print(f"✓ Saved: combined_dataset.csv ({df_combined.shape[0]} rows)")

# Also save as Excel for easier viewing
try:
    df_combined.to_excel('combined_dataset.xlsx', index=False, sheet_name='Vitals')
    print(f"✓ Saved: combined_dataset.xlsx")
except:
    print("⚠ Excel save skipped (openpyxl not installed)")

print("\n" + "=" * 60)
print("STEP 5: NORMALIZE VITAL SIGNS (for model input)")
print("=" * 60)

# Vital signs that go INTO the model (will be normalized)
vital_cols = ['hr', 'hrv', 'spo2', 'bp_s', 'bp_d']
scaler = StandardScaler()

df_combined[vital_cols] = scaler.fit_transform(df_combined[vital_cols])
print(f"✓ Vital signs normalized: {vital_cols}")

# Keep age and activity UNNORMALIZED (for context/reason codes)
print(f"✓ Age and activity kept as context (NOT normalized)")

print("\n" + "=" * 60)
print("STEP 6: SAVE CONTEXT DATA (age + activity for reason codes)")
print("=" * 60)

# Save age and activity separately for use in reason-code generation
context_data = df_combined[['age', 'activity']].values  # Shape: (10720, 2)
np.save('context_data.npy', context_data)
print(f"✓ Saved: context_data.npy")
print(f"  Shape: {context_data.shape}")
print(f"  Use this to add age/activity context to reason codes later")

print("\n" + "=" * 60)
print("STEP 7: CREATE TIME-SERIES WINDOWS (60 seconds, 5 features)")
print("=" * 60)

window_length = 60
feature_cols = ['hr', 'hrv', 'spo2', 'bp_s', 'bp_d']
num_features = len(feature_cols)

windows = []
labels = []

for idx, row in df_combined.iterrows():
    # Base vital signs (normalized)
    hr_base = row['hr']
    hrv_base = row['hrv']
    spo2_base = row['spo2']
    bp_s_base = row['bp_s']
    bp_d_base = row['bp_d']
    activity = row['activity']
    label = row['target']
    
    # Create a 60-second window
    window = []
    
    for second in range(window_length):
        # Noise based on activity
        if activity == 0:  # Resting
            hr_noise = np.random.normal(0, 0.05)
            hrv_noise = np.random.normal(0, 0.03)
            spo2_noise = np.random.normal(0, 0.02)
            bp_s_noise = np.random.normal(0, 0.02)
            bp_d_noise = np.random.normal(0, 0.02)
        elif activity == 1:  # Walking
            hr_noise = np.random.normal(0, 0.10)
            hrv_noise = np.random.normal(0, 0.08)
            spo2_noise = np.random.normal(0, 0.05)
            bp_s_noise = np.random.normal(0, 0.08)
            bp_d_noise = np.random.normal(0, 0.08)
        else:  # Running
            hr_noise = np.random.normal(0, 0.15)
            hrv_noise = np.random.normal(0, 0.12)
            spo2_noise = np.random.normal(0, 0.08)
            bp_s_noise = np.random.normal(0, 0.12)
            bp_d_noise = np.random.normal(0, 0.12)
        
        # Create this second's data
        hr_t = np.clip(hr_base + hr_noise, -3, 3)
        hrv_t = np.clip(hrv_base + hrv_noise, -3, 3)
        spo2_t = np.clip(spo2_base + spo2_noise, -3, 3)
        bp_s_t = np.clip(bp_s_base + bp_s_noise, -3, 3)
        bp_d_t = np.clip(bp_d_base + bp_d_noise, -3, 3)
        
        window.append([hr_t, hrv_t, spo2_t, bp_s_t, bp_d_t])
    
    windows.append(window)
    labels.append(label)
    
    if (idx + 1) % 2000 == 0:
        print(f"  Processed {idx + 1}/{len(df_combined)} samples...")

windows_array = np.array(windows)
labels_array = np.array(labels)

print(f"\n✓ Windows created!")
print(f"  Shape: {windows_array.shape}")
print(f"    - {windows_array.shape[0]} time-series windows")
print(f"    - {windows_array.shape[1]} seconds per window")
print(f"    - {windows_array.shape[2]} features per second [HR, HRV, SpO2, BP_S, BP_D]")

print("\n" + "=" * 60)
print("STEP 8: SAVE ALL PROCESSED DATA")
print("=" * 60)

# Save windows and labels
np.save('windows_combined.npy', windows_array)
np.save('labels_combined.npy', labels_array)

print(f"✓ Saved:")
print(f"  - windows_combined.npy (model input)")
print(f"  - labels_combined.npy (labels)")
print(f"  - context_data.npy (age + activity for reason codes)")
print(f"  - combined_dataset.csv (original data)")
print(f"  - combined_dataset.xlsx (Excel view)")

print("\n" + "=" * 60)
print("STEP 9: VERIFY - SAMPLE WINDOWS")
print("=" * 60)

print(f"\nExample 1: First window (60 seconds)")
print(f"  Features: [HR, HRV, SpO2, BP_S, BP_D]")
print(f"  First 5 seconds:")
print(windows_array[0][:5])
print(f"  Context: age={context_data[0][0]:.0f}, activity={context_data[0][1]:.0f}")
print(f"  Label: {labels_array[0]}")

print(f"\nExample 2: Random window")
random_idx = np.random.randint(0, len(windows_array))
print(f"  Window {random_idx}, First 5 seconds:")
print(windows_array[random_idx][:5])
print(f"  Context: age={context_data[random_idx][0]:.0f}, activity={context_data[random_idx][1]:.0f}")
print(f"  Label: {labels_array[random_idx]}")

print("\n" + "=" * 60)
print("✓ DATASET READY FOR TRAINING")
print("=" * 60)
print(f"\nFinal dataset:")
print(f"  - {windows_array.shape[0]} time-series windows")
print(f"  - Each: 60 seconds × 5 features [HR, HRV, SpO2, BP_S, BP_D]")
print(f"  - Context available: age + activity (for reason codes)")
print(f"  - Labels: {np.bincount(labels_array)}")
print(f"\nFiles saved:")
print(f"  1. windows_combined.npy ← use for model training")
print(f"  2. labels_combined.npy ← use for model training")
print(f"  3. context_data.npy ← use for reason-code generation")
print(f"  4. combined_dataset.csv ← inspect/share")
print(f"  5. combined_dataset.xlsx ← view in Excel")
print(f"\nNext step: Train neural network with Federated Learning")
