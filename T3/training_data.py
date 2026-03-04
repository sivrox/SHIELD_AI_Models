import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("🛡️ S.H.I.E.L.D. DATA PROCESSOR: FULL SPECTRUM PIPELINE")
print("=" * 60)

# --- STEP 1: LOAD 100% OF BOTH DATASETS ---
# Reasoning: We use all available data to ensure the LSTM captures 
# as many unique cardiovascular signatures as possible.
df_real = pd.read_csv('T3/combined_datasets/final_vitals.csv')
df_synthetic = pd.read_csv('T3/combined_datasets/synthetic_vitals.csv')

# Combine all rows from both sources
df_combined = pd.concat([df_real, df_synthetic], ignore_index=True) 

# Shuffle immediately to mix real and synthetic patterns
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Total Combined Rows: {df_combined.shape[0]}")
print(f"  - Clinical (Real): {len(df_real)} | Simulated (Synthetic): {len(df_synthetic)}")

# --- STEP 2: INFER ACTIVITY & CLEAN LABELS ---
def infer_activity(row):
    """Refined activity mapping based on physiological markers."""
    hr, hrv, current = row['hr'], row['hrv'], row['activity']
    if current != 0: return current
    if hr > 110 and hrv < 30: return 2   # Exercise/High Stress
    elif hr > 85 and hrv < 50: return 1  # Moderate Activity
    return 0                            # Resting

df_combined['activity'] = df_combined.apply(infer_activity, axis=1)

# Add Sleep if missing (Clinical baseline: Healthy ~7.5h, Risk ~5.5h)
if 'sleep' not in df_combined.columns:
    np.random.seed(42)
    df_combined['sleep'] = np.where(df_combined['target'] == 0, 
                                    np.random.normal(7.5, 0.7, len(df_combined)), 
                                    np.random.normal(5.5, 1.0, len(df_combined)))
    df_combined['sleep'] = df_combined['sleep'].clip(3, 10).round(1)

# Ensure Target is strictly Integer for the training logic
df_combined['target'] = df_combined['target'].fillna(0).astype(int)

# --- STEP 3: EXPORT RAW CSV FOR HUMAN VISUALIZATION ---
# Reasoning: We save this BEFORE scaling so you can see real units 
# (e.g., HR 72, SpO2 98) in your Excel/Tableau reports.
vital_cols = ['hr', 'hrv', 'spo2', 'bp_s', 'bp_d']
context_cols = ['age', 'activity', 'sleep']
ordered_cols = vital_cols + context_cols + ['target']

df_combined[ordered_cols].to_csv('T3/combined_training_data.csv', index=False)
print(f"✓ Saved: combined_training_data.csv (Human-Readable Original Units)")

# --- STEP 4: SCALE VITALS FOR AI PROCESSING ---
# Reasoning: StandardScaler centers data at 0. This is for the AI's math, 
# not for human eyes. It prevents high-number columns (like BP) from 
# dominating low-number columns (like SpO2).
scaler = StandardScaler()
df_combined[vital_cols] = scaler.fit_transform(df_combined[vital_cols])
print(f"✓ AI Scaling Applied to: {vital_cols}")

# --- STEP 5: TEMPORAL WINDOWING (60s Windows) ---
print("\n🎬 Creating Time-Series Window (60s windows)...")
window_length = 60
num_features = len(vital_cols)

windows = []
labels = []
context_list = []

for idx, row in df_combined.iterrows():
    vitals_base = row[vital_cols].values
    activity = row['activity']
    label = row['target']
    context = row[context_cols].values
    
    # Generate 60 seconds of noisy time-series data from one baseline row
    noise_level = 0.05 if activity == 0 else 0.10 if activity == 1 else 0.15
    noise = np.random.normal(0, noise_level, size=(window_length, num_features))
    
    # Window shape: (60, 5) - uses scaled numbers from Step 4
    window = np.clip(vitals_base + noise, -3, 3)
    
    windows.append(window)
    labels.append(label)
    context_list.append(context)

X_windows = np.array(windows) # (Samples, 60, 5)
y_train = np.array(labels)    # (Samples,)
X_context = np.array(context_list) # (Samples, 3)

# --- STEP 6: THE FINAL INTEGRATION (The 'Glue' Phase) ---
print("\n🏗️  Merging Vitals and Context into Final Training Tensor...")

# Repeat the static context (Age, Activity, Sleep) 60 times to match the timeline
context_expanded = np.repeat(X_context[:, np.newaxis, :], window_length, axis=1)

# Concatenate along the feature axis (axis 2)
# Resulting Shape: (Samples, 60, 8)
X_train = np.concatenate([X_windows, context_expanded], axis=2)

# --- STEP 7: FINAL EXPORT ---
np.save('T3/X_train.npy', X_train)
np.save('T3/y_train.npy', y_train)

print("=" * 60)
print(f"✅ SUCCESS: DATA PIPELINE COMPLETE")
print(f"   -> Final X_train Shape: {X_train.shape} (8 Features)")
print(f"   -> Final y_train Shape: {y_train.shape}")
print("=" * 60)