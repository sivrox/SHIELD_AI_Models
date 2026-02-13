import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import shap

# 1. DATA PREPARATION
data = pd.read_csv('combined_dataset.csv')
print(f"CSV Columns: {data.columns.tolist()}")

# Define feature list
features_in_csv = ['hr', 'bp_s', 'hrv', 'bp_d', 'spo2', 'activity', 'age', 'sleep']
X = data[features_in_csv].copy()
y = data['target']

# Mapping activities (Handles both Strings and Numbers)
activity_map = {
    'SLEEP': 0,    # We will treat 0 as the lowest state
    'REST': 0,     # Your data's 0.0 (HR 93)
    'ACTIVE': 1,   # Your data's 1.0 (HR 104)
    'EXERCISE': 2  # Your data's 2.0 (HR 105)
}

# Since your CSV is already numeric, we need to make sure 
# it fits the 0, 1, 2 structure
X['activity'] = X['activity'].fillna(0).astype(int)

# Check if 'activity' is strings or numbers
if X['activity'].dtype == object:
    print("Detected string activities. Mapping to numbers...")
    X['activity'] = X['activity'].str.upper().map(activity_map)
else:
    print("Detected numeric activities. Skipping string mapping...")

X['activity'] = X['activity'].fillna(1).astype(int)

# Calculate Stats for Export
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

# Scale Numerical Features
X_scaled = X.copy()
for col in features_in_csv:
    if col != 'activity':
        X_scaled[col] = (X[col] - stats[col]['mean']) / stats[col]['std']

# SMOTE to balance data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Create 60-second Sequences (8 Features)
X_seq = np.repeat(X_res.values[:, np.newaxis, :], 60, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_res, test_size=0.2)

# 2. MODEL
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(60, 8)), 
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining with 8 features...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 3. SHAP & EXPORT
print("\nCalculating SHAP importance...")
background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(X_test[:50])
shap_data = shap_values[0] if isinstance(shap_values, list) else shap_values
mean_shap = np.abs(shap_data).mean(axis=(0, 1))

shield_weights = {features_in_csv[i]: float(mean_shap[i]) for i in range(len(features_in_csv))}
shield_weights['activity_level'] = shield_weights.pop('activity')

# Save all files
with open('shield_weights.json', 'w') as f: json.dump(shield_weights, f, indent=2)
with open('scaler_stats.json', 'w') as f: json.dump(stats, f, indent=2)

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 60, 8], model.inputs[0].dtype))
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

with open('shield_v3.tflite', 'wb') as f: f.write(tflite_model)

print("\nDeployment files ready: shield_v3.tflite, shield_weights.json, scaler_stats.json")
