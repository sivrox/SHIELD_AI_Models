'''
We utilized Hybrid Synthetic Augmentation.
Public clinical datasets like UCI are high-quality but 'Legacy'—they lack
the multi-modal metrics provided by modern IoT sensors like HRV and SpO2.
To train a 'Digital Twin' that functions in a modern wearable ecosystem,
we augmented the real clinical 'Anchors' (Age, BP, Diagnosis) with
synthetic features that follow established medical correlations.
This ensures our model learns the inter-dependency of these metrics,
which is the core innovation of S.H.I.E.L.D.
'''

import pandas as pd
import numpy as np
import wfdb
import os

# --- PART 1: STACKING ALL UCI DATASETS ---
def get_uci_data(folder_path):
    files = [
        'processed.cleveland.data', 
        'processed.hungarian.data', 
        'processed.switzerland.data', 
        'processed.long_beach_va.data'
    ]
    
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    all_dfs = []
    for file in files:
        full_path = os.path.join(folder_path, file)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path, names=columns, na_values="?")
            all_dfs.append(df)
            print(f"Loaded {file}: {len(df)} patients found.")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Selection of core metrics
    final_df = combined_df[['age', 'thalach', 'trestbps', 'target']].copy()
    final_df.columns = ['age', 'hr', 'bp_s', 'target']
    
    # Fill gaps with column averages
    final_df = final_df.fillna(final_df.mean())
    return final_df

# --- PART 2: THE MIT-BIH HRV EXTRACTOR ---
def get_mit_vitals(folder_path):
    records = [f.replace('.hea', '') for f in os.listdir(folder_path) if f.endswith('.hea')]
    
    mit_data = []
    for r in records:
        try:
            path = os.path.join(folder_path, r)
            record = wfdb.rdrecord(path)
            ann = wfdb.rdann(path, 'atr')
            heartbeat_times = ann.sample / record.fs
            rr_intervals = np.diff(heartbeat_times)
            hrv = np.std(rr_intervals) * 1000 
            avg_hr = 60 / np.mean(rr_intervals)
            mit_data.append({'hr': avg_hr, 'hrv': hrv})
        except Exception as e:
            print(f"Skipping record {r} due to error: {e}")
            
    return pd.DataFrame(mit_data)

# --- PART 3: THE FINAL MERGE ---
def create_combined_csv():
    print("Starting Data Ingestion...")
    np.random.seed(42) 
    
    # 1. Process UCI 
    uci_df = get_uci_data('uci_data')
    
    # 2. Extract real-world averages for baseline context
    mit_df = get_mit_vitals('mit_bih_data')
    avg_hrv_mit = mit_df['hrv'].mean() # We'll use this as a reference
    
    # 3. Clean and Simplfy Target
    # 0 = Healthy, 1 = Risk
    uci_df['target'] = (uci_df['target'] > 0).astype(int)
    
    # --- 4. SMART SYNTHETIC GENERATION (Target-Based) ---
    # Why: Healthy people should have better vitals than sick people.
    
    # HRV Generation
    # Healthy (target 0): High HRV (65 +/- 15)
    # Risk (target 1): Low HRV (30 +/- 10)
    uci_df['hrv'] = np.where(
        uci_df['target'] == 0,
        np.random.normal(65, 15, size=len(uci_df)),
        np.random.normal(30, 10, size=len(uci_df))
    )
    uci_df['hrv'] = uci_df['hrv'].clip(lower=15, upper=110).round(1)
    
    # BP_D (Diastolic) Generation with noise
    noise = np.random.uniform(0.9, 1.1, size=len(uci_df))
    uci_df['bp_d'] = (uci_df['bp_s'] * 0.68 * noise).round(1)
    
    # SpO2 Generation
    # Healthy (target 0): 97-99%
    # Risk (target 1): 92-96%
    uci_df['spo2'] = np.where(
        uci_df['target'] == 0,
        np.random.randint(97, 100, size=len(uci_df)),
        np.random.randint(92, 97, size=len(uci_df))
    )
    
    # Activity (Assume resting for this clinical baseline)
    uci_df['activity'] = 0 
    
    # --- 5. FINAL REORDER & SAVE ---
    # Force 'target' to be the very last column
    cols = [c for c in uci_df.columns if c != 'target'] + ['target']
    uci_df = uci_df[cols]
    
    uci_df.to_csv('final_vitals.csv', index=False)
    
    print(f"\nSUCCESS! Created 'final_vitals.csv' with {len(uci_df)} patients.")
    print("AI-Ready Columns: ", list(uci_df.columns))
    print("\nSample Data Quality Check (First 5 rows):")
    print(uci_df.head())

if __name__ == "__main__":
    create_combined_csv()