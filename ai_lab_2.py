import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib # Used to save our 'Equalizer' for the app

# --- STEP 1: LOADING & ALIGNING ---
def load_data():
    public_df = pd.read_csv('public_vitals.csv')
    synthetic_df = pd.read_csv('synthetic_vitals.csv')
    
    cols = ['age', 'hr', 'bp_s', 'hrv', 'bp_d', 'spo2', 'activity', 'target']
    df = pd.concat([public_df[cols], synthetic_df[cols]], ignore_index=True)
    
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- STEP 2: PREPROCESSING ---
def preprocess_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'shield_scaler.pkl')
    return X_train_scaled, X_val_scaled, X_test_scaled

# --- STEP 3: HYPERPARAMETER TUNING ---
def build_tunable_model(hp):
    model = tf.keras.Sequential()
    hp_units = hp.Int('units', min_value=16, max_value=64, step=16)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu', input_shape=(7,)))
    
    hp_dropout = hp.Float('dropout', 0.1, 0.3, step=0.1)
    model.add(tf.keras.layers.Dropout(hp_dropout))
    
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- EXECUTION START ---
X_train, X_val, X_test, y_train, y_val, y_test = load_data()
X_train_s, X_val_s, X_test_s = preprocess_data(X_train, X_val, X_test)

tuner = kt.Hyperband(
    build_tunable_model,
    objective='val_accuracy',
    max_epochs=20,
    directory='tuning_lab',
    project_name='shield_v1'
)

print("\n--- PHASE 1: TUNING ---")
tuner.search(X_train_s, y_train, epochs=20, validation_data=(X_val_s, y_val))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# --- STEP 4: FINAL TRAINING ---
print("\n--- PHASE 2: FINAL TRAINING ---")
model = tuner.hypermodel.build(best_hps)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model.fit(X_train_s, y_train, epochs=50, validation_data=(X_val_s, y_val), callbacks=[stop_early])

# --- STEP 5: PROFESSIONAL EVALUATION (The 'Grad Project' Part) ---
print("\n--- PHASE 3: GRADUATION PERFORMANCE REPORT ---")
# 1. Get Predictions on the Test Set (The 'Final Exam')
y_pred_prob = model.predict(X_test_s)
y_pred = (y_pred_prob > 0.5).astype(int) # Threshold at 50%

# 2. Print Classification Report
# This shows Precision (True Alarms) and Recall (Caught Emergencies)
print("\n[Metric Report]")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'At-Risk']))

# 3. Print Confusion Matrix
# Shows where the AI 'Got Confused'
print("\n[Confusion Matrix]")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives (Correctly Healthy): {cm[0][0]}")
print(f"False Positives (False Alarms): {cm[0][1]}")
print(f"False Negatives (Missed Risk!): {cm[1][0]}")
print(f"True Positives (Correctly At-Risk): {cm[1][1]}")

# --- STEP 6: REASON CODE GENERATOR (Edge-XAI Logic) ---
def get_reason_code(patient_data_scaled, original_data, scaler):
    """
    Simulates XAI by finding which feature deviated most from the norm.
    """
    # Features names for the report
    features = ['age', 'hr', 'bp_s', 'hrv', 'bp_d', 'spo2', 'activity']
    
    # Calculate 'Abnormality Score' (distance from the mean 0)
    # Since data is scaled (mean=0, std=1), the largest absolute value 
    # is the biggest driver of the risk.
    abnormality = np.abs(patient_data_scaled[0])
    top_feature_index = np.argmax(abnormality)
    
    reason = features[top_feature_index].upper()
    return f"REASON_{reason}"

# Test XAI on a random high-risk patient from the test set
sample_idx = 0 
sample_data = X_test_s[sample_idx:sample_idx+1]
sample_score = model.predict(sample_data)[0][0]
print(f"\n--- SAMPLE PREDICTION TEST ---")
print(f"Risk Score: {sample_score*100:.1f}%")
print(f"XAI Reason Code: {get_reason_code(sample_data, X_test[sample_idx], None)}")

# --- STEP 7: SAVE FOR MOBILE ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('shield_v1.tflite', 'wb') as f:
    f.write(tflite_model)
print("\nDeployment file 'shield_v1.tflite' and 'shield_scaler.pkl' are ready.")