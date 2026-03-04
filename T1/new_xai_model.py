import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import shap

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================
data = pd.read_csv('T1/combined_dataset.csv')
print(f"CSV Columns: {data.columns.tolist()}")

# Sort by time to preserve temporal order for sequence building
if 'timestamp' in data.columns:
    data = data.sort_values('timestamp').reset_index(drop=True)
    print("Data sorted by timestamp.")
else:
    print("WARNING: No 'timestamp' column found. Assuming data is already time-ordered.")

# Define feature list
features_in_csv = ['hr', 'bp_s', 'hrv', 'bp_d', 'spo2', 'activity', 'age', 'sleep']
X = data[features_in_csv].copy()
y = data['target'].values

# Mapping activities (handles both strings and numbers)
activity_map = {
    'SLEEP': 0,
    'REST': 0,
    'ACTIVE': 1,
    'EXERCISE': 2
}

X['activity'] = X['activity'].fillna(0)
if X['activity'].dtype == object:
    print("Detected string activities. Mapping to numbers...")
    X['activity'] = X['activity'].str.upper().map(activity_map)
else:
    print("Detected numeric activities. Skipping string mapping...")

X['activity'] = X['activity'].fillna(1).astype(int)

# =============================================================================
# 2. COMPUTE SCALER STATS (on raw data, before sequencing)
# =============================================================================
stats = {}
for col in features_in_csv:
    if col == 'activity':
        stats['activity_level'] = {"mean": 0.0, "std": 1.0, "type": "categorical"}
    else:
        stats[col] = {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "type": "numerical"
        }

# Scale numerical features
X_scaled = X.copy()
for col in features_in_csv:
    if col != 'activity':
        X_scaled[col] = (X[col] - stats[col]['mean']) / stats[col]['std']

# =============================================================================
# 3. BUILD REAL SLIDING-WINDOW SEQUENCES
# FIX: Replace np.repeat (identical timesteps) with genuine temporal windows.
# Each sequence captures 60 consecutive timesteps of real physiological change.
# =============================================================================
WINDOW_SIZE = 60

def create_sequences(X_arr, y_arr, window_size=60, step=1):
    """
    Slide a window over time-ordered data to produce (window_size, n_features)
    sequences. The label assigned to each sequence is the target value at the
    timestep immediately following the window.
    """
    Xs, ys = [], []
    for i in range(0, len(X_arr) - window_size, step):
        Xs.append(X_arr[i : i + window_size])
        ys.append(y_arr[i + window_size])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled.values, y, window_size=WINDOW_SIZE)
print(f"\nSequence shape: {X_seq.shape}  —  {X_seq.shape[0]} samples, "
      f"{X_seq.shape[1]} timesteps, {X_seq.shape[2]} features")

# =============================================================================
# 4. SMOTE — applied AFTER sequence creation to avoid label leakage
# Flatten sequences → resample → reshape back
# =============================================================================
n_samples, n_steps, n_features = X_seq.shape
X_flat = X_seq.reshape(n_samples, n_steps * n_features)

smote = SMOTE(random_state=42)
X_res_flat, y_res = smote.fit_resample(X_flat, y_seq)
print(f"After SMOTE — samples: {X_res_flat.shape[0]}, "
      f"class balance: {np.bincount(y_res.astype(int))}")

X_res = X_res_flat.reshape(-1, n_steps, n_features)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# =============================================================================
# 5. DIAGNOSTICS — catch label/data issues before training
# =============================================================================
print(f"\nOriginal rows: {len(data)}, Sequences generated: {len(X_seq)}")
print(f"Stress rate in original data: {data['target'].mean():.2%}")
print(f"Class balance after SMOTE: {np.bincount(y_res.astype(int))}")

# =============================================================================
# 6. MODEL — stacked LSTM with wider dense layers
# =============================================================================
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(WINDOW_SIZE, n_features)),
    tf.keras.layers.LSTM(64, return_sequences=True),   # stacked: first layer keeps sequence
    tf.keras.layers.LSTM(32, return_sequences=False),  # stacked: second layer collapses
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Early stopping + learning rate reduction — allows up to 50 epochs but stops when
# val_loss stops improving, and halves LR when it plateaus
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
    )
]

print("\nTraining with real temporal sequences (up to 50 epochs w/ early stopping)...")
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks
)

# =============================================================================
# 6. EVALUATION — explicit threshold assessment
# =============================================================================
print("\nEvaluating model...")
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)
print(classification_report(y_test.astype(int), y_pred,
                             target_names=['No Stress', 'Stress']))

# =============================================================================
# 7. SHAP EXPLAINABILITY
# =============================================================================
print("\nCalculating SHAP importance...")
background_idx = np.random.choice(X_train.shape[0], 50, replace=False)
background = X_train[background_idx]

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_test[:50])

# shap_values shape: (samples, timesteps, features) — average over time axis
shap_data = shap_values[0] if isinstance(shap_values, list) else shap_values
mean_shap = np.abs(shap_data).mean(axis=(0, 1))  # (n_features,)

shield_weights = {features_in_csv[i]: float(mean_shap[i]) for i in range(len(features_in_csv))}
shield_weights['activity_level'] = shield_weights.pop('activity')

print("\nSHAP feature importance:")
for feat, val in sorted(shield_weights.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {val:.4f}")

# =============================================================================
# 8. EXPORT DEPLOYMENT FILES
# =============================================================================
with open('T1/shield_weights.json', 'w') as f:
    json.dump(shield_weights, f, indent=2)

with open('T1/scaler_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, WINDOW_SIZE, n_features], model.inputs[0].dtype)
)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

with open('T1/shield_v3.tflite', 'wb') as f:
    f.write(tflite_model)

print("\nDeployment files ready: shield_v3.tflite, shield_weights.json, scaler_stats.json")