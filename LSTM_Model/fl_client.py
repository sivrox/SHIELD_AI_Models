"""
S.H.I.E.L.D. Federated Learning Client
=======================================
Simulates an FL client (hospital/device) that:
  1. Loads a partition of patient data
  2. Downloads the current global model from the FL server
  3. Trains locally on its own data
  4. Sends the updated weights back to the server

Usage:
  python fl_client.py --client-id hospital_0 --data shield_dataset.csv --hospital 0
  python fl_client.py --client-id hospital_1 --data shield_dataset.csv --hospital 1
  python fl_client.py --client-id hospital_2 --data shield_dataset.csv --hospital 2

For demo: run the server first, then run 2+ clients in separate terminals.
Requirements: pip install tensorflow numpy requests pandas
"""

import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import sys

# === CONFIG ===
server_url = "http://localhost:8080"
feature_order = ['hr', 'bp_s', 'hrv', 'bp_d', 'spo2', 'activity', 'age', 'sleep']
target_col = 'stress_score'
window_size = 60
local_epochs = 3
batch_size = 64


# === Model (must match server and training pipeline exactly) ===

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(window_size, 8)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# === Data Loading ===

def load_hospital_data(csv_path, hospital_id):
    """Load and preprocess data for one hospital partition."""
    print(f"  Loading data for hospital {hospital_id}...")
    df = pd.read_csv(csv_path)
    df_hospital = df[df['hospital_id'] == hospital_id].copy()

    if len(df_hospital) == 0:
        print(f"  ERROR: No data found for hospital {hospital_id}")
        sys.exit(1)

    n_patients = df_hospital['patient_id'].nunique()
    print(f"  Found {n_patients} patients, {len(df_hospital):,} rows")

    # Load scaler stats if available, otherwise compute from this partition
    try:
        with open('outputs/scaler_stats.json', 'r') as f:
            scaler = json.load(f)
    except FileNotFoundError:
        print("  WARNING: scaler_stats.json not found, computing from local data")
        scaler = {}
        for col in feature_order:
            if col == 'activity':
                scaler['activity_level'] = {"mean": 0, "std": 1, "type": "categorical"}
            else:
                scaler[col] = {"mean": float(df[col].mean()),
                               "std": float(df[col].std()), "type": "numerical"}

    # Normalize
    for col in feature_order:
        if col != 'activity':
            df_hospital[col] = (df_hospital[col] - scaler[col]['mean']) / scaler[col]['std']

    # Build sequences
    seqs, labels = [], []
    for pid in df_hospital['patient_id'].unique():
        patient = df_hospital[df_hospital['patient_id'] == pid].sort_values('timestamp')
        features = patient[feature_order].values
        targets = patient[target_col].values
        for i in range(0, len(features) - window_size, 30):
            seqs.append(features[i:i + window_size])
            labels.append(targets[i + window_size - 1])

    X = np.array(seqs, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    print(f"  Built {len(X):,} training sequences")
    return X, y


# === FL Client Logic ===

def download_global_weights():
    """Fetch current global model weights from the FL server."""
    try:
        resp = requests.get(f"{server_url}/fl/global-weights", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        weights = [np.array(w, dtype=np.float32) for w in data['weights']]
        print(f"  Downloaded global weights (round {data['round']})")
        return weights
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to FL server at {server_url}")
        print(f"  Make sure fl_server.py is running first!")
        sys.exit(1)

def train_locally(model, X, y):
    """Train the model on local hospital data."""
    print(f"  \nTraining locally for {local_epochs} epochs...")
    history = model.fit(X, y, epochs=local_epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=1)
    final_loss = history.history['loss'][-1]
    final_mae = history.history['mae'][-1]
    print(f"  Local training done — Loss: {final_loss:.4f}, MAE: {final_mae:.4f}")
    return model

def submit_weights(model, n_samples, client_id):
    """Send locally-trained weights to the FL server."""
    weights_list = [w.tolist() for w in model.get_weights()]
    payload = {
        "client_id": client_id,
        "n_samples": n_samples,
        "weights": weights_list
    }
    print(f"  Submitting weights to server ({n_samples} samples)...")
    resp = requests.post(f"{server_url}/fl/submit", json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    print(f"  Server response: {result['message']}")
    return result


# === Main ===

def main():
    global server_url
    parser = argparse.ArgumentParser(description="S.H.I.E.L.D. FL Client")
    parser.add_argument('--client-id', type=str, required=True,
                        help='Unique client identifier (e.g. hospital_0)')
    parser.add_argument('--data', type=str, default='shield_dataset.csv',
                        help='Path to the dataset CSV')
    parser.add_argument('--hospital', type=int, required=True,
                        help='Hospital ID to filter data (0-4)')
    parser.add_argument('--server', type=str, default=server_url,
                        help='FL server URL')
    args = parser.parse_args()

    server_url = args.server

    print(f"\nS.H.I.E.L.D FL Client: {args.client_id}\n")

    # Step 1: Load local hospital data
    X, y = load_hospital_data(args.data, args.hospital)

    # Step 2: Download current global model
    print("\n[1/3] Downloading global model from server...")
    global_weights = download_global_weights()

    # Step 3: Initialize local model with global weights and train
    print("\n[2/3] Training on local data...")
    local_model = build_model()
    local_model.set_weights(global_weights)
    local_model = train_locally(local_model, X, y)

    # Step 4: Submit updated weights to server
    print("\n[3/3] Submitting weights to FL server...")
    result = submit_weights(local_model, len(X), args.client_id)

    print(f"\nClient {args.client_id} - FL round complete")

if __name__ == "__main__":
    main()