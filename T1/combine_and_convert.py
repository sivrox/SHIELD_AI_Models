import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("STEP 1: LOAD BOTH DATASETS")
print("=" * 60)

# Load real dataset
df_real = pd.read_csv('combined_datasets/final_vitals.csv')
print(f"✓ Real dataset loaded: {df_real.shape[0]} rows")

# Load synthetic dataset
df_synthetic = pd.read_csv('combined_datasets/synthetic_vitals.csv')
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

# --- ADDED: SLEEP FEATURE & LABEL CLEANING ---
if 'sleep' not in df_combined.columns:
    print("-> Adding Sleep Duration feature (Hours)...")
    np.random.seed(42)
    # Healthy people (0) get ~7.5 hrs, At-risk (1) get ~5.5 hrs
    df_combined['sleep'] = np.where(df_combined['target'] == 0, 
                                    np.random.normal(7.5, 0.7, len(df_combined)), 
                                    np.random.normal(5.5, 1.0, len(df_combined)))
    df_combined['sleep'] = df_combined['sleep'].clip(3, 10).round(1)

# Fix for the 'np.bincount' TypeError: Force target to be integer
df_combined['target'] = df_combined['target'].fillna(0).astype(int)

print("\n" + "=" * 60)
print("STEP 4: SAVE COMBINED DATASET (before normalization)")
print("=" * 60)

# Save the combined dataset for inspection
df_combined.to_csv('combined_dataset.csv', index=False)
print(f"✓ Saved: combined_dataset.csv ({df_combined.shape[0]} rows)")

print("\n" + "=" * 60)
print("STEP 5: NORMALIZE VITAL SIGNS (for model input)")
print("=" * 60)

# Vital signs that go INTO the model (will be normalized)
vital_cols = ['hr', 'hrv', 'spo2', 'bp_s', 'bp_d']
scaler = StandardScaler()

df_combined[vital_cols] = scaler.fit_transform(df_combined[vital_cols])
print(f"✓ Vital signs normalized: {vital_cols}")

print("\n" + "=" * 60)
print("STEP 6: SAVE CONTEXT DATA (age + activity + sleep for reason codes)")
print("=" * 60)

# Updated to include 'sleep' in the context data
context_data = df_combined[['age', 'activity', 'sleep']].values
np.save('X_context.npy', context_data)
print(f"✓ Saved: X_context.npy")
print(f"  Shape: {context_data.shape}")

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
    vitals_base = row[feature_cols].values
    activity = row['activity']
    label = row['target']
    
    window = []
    
    for second in range(window_length):
        # Noise based on activity
        if activity == 0:  # Resting
            noise_level = 0.05
        elif activity == 1:  # Walking
            noise_level = 0.10
        else:  # Running
            noise_level = 0.15
            
        noise = np.random.normal(0, noise_level, size=num_features)
        
        # Create this second's data
        v_t = np.clip(vitals_base + noise, -3, 3)
        window.append(v_t)
    
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

# Save windows and labels using your specific naming convention
np.save('X_windows.npy', windows_array)
np.save('y_labels.npy', labels_array.astype(int)) # Safety cast to int here too

print(f"✓ Saved:")
print(f"  - X_windows.npy (model input)")
print(f"  - y_labels.npy (labels)")
print(f"  - X_context.npy (age + activity + sleep for reason codes)")
print(f"  - combined_dataset.csv (original data)")

print("\n" + "=" * 60)
print("STEP 9: VERIFY - SAMPLE WINDOWS")
print("=" * 60)

print(f"\nExample 1: First window (60 seconds)")
print(f"  Features: [HR, HRV, SpO2, BP_S, BP_D]")
print(f"  First 5 seconds:")
print(windows_array[0][:5])
# Show age, activity, and sleep in context
print(f"  Context: age={context_data[0][0]:.0f}, activity={context_data[0][1]:.0f}, sleep={context_data[0][2]:.1f}h")
print(f"  Label: {labels_array[0]}")

print(f"\nExample 2: Random window")
random_idx = np.random.randint(0, len(windows_array))
print(f"  Window {random_idx}, First 5 seconds:")
print(windows_array[random_idx][:5])
print(f"  Context: age={context_data[random_idx][0]:.0f}, activity={context_data[random_idx][1]:.0f}, sleep={context_data[random_idx][2]:.1f}h")
print(f"  Label: {labels_array[random_idx]}")

print("\n" + "=" * 60)
print("✓ DATASET READY FOR TRAINING")
print("=" * 60)
print(f"\nFinal dataset stats:")
print(f"  - Total Windows: {len(labels_array)}")
# This line works perfectly now because of the casting in Step 3
print(f"  - Class Balance: {np.bincount(labels_array.astype(int))}") 
print(f"\nNext step: Train neural network with Federated Learning")