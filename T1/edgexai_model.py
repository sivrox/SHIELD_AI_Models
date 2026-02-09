import json
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import shap
import joblib

# # --- STEP 1: DATA INGESTION & IMPUTATION ---
# def load_and_impute():
#     print("📥 Loading windows, labels, and context...")
#     X = np.load('X_windows.npy')    # (Samples, 60, 5)
#     y = np.load('y_labels.npy')     # (Samples,)
#     context = np.load('X_context.npy') # (Samples, 3) - Age, Activity, Sleep
    
#     # IMPUTATION: If there are any NaNs (missing data), we fill them with 0 (The Mean)
#     # This ensures the model doesn't crash during live streaming.
#     X = np.nan_to_num(X, nan=0.0)
    
#     return X, y, context

# --- STEP 1: CORRECTED DATA INGESTION ---
def load_and_impute():
    print("📥 Loading windows, labels, and context...")
    
    # Load raw data
    X = np.load('X_windows.npy')    # Expected: (Samples, 60, 5)
    y = np.load('y_labels.npy')     # (Samples,)
    context = np.load('X_context.npy') # (Samples, 3)
    
    # 1. Verification of Vitals Order
    # The App expects: [HR, HRV, SPO2, BP_S, BP_D]
    # Ensure your window creation script used these indices: [1, 3, 5, 2, 4]
    
    if X.shape[2] != 5:
        raise ValueError(f"CRITICAL: Model expects 5 vitals, but found {X.shape[2]}. "
                         "Check your window creation script!")

    # 2. IMPUTATION
    # We use 0.0 because the data is likely already Z-Score normalized 
    # (where 0 is the mean). If not normalized, use np.nanmean().
    X = np.nan_to_num(X, nan=0.0)
    context = np.nan_to_num(context, nan=0.0)
    
    print(f"✅ Data loaded. Window Shape: {X.shape}")
    print(f"📊 Features assumed: [HR, HRV, SPO2, BP_S, BP_D]")
    
    return X, y, context

# --- STEP 2: HYPER-PARAMETER TUNING (The "Search for the Best Brain") ---
def build_model(hp):
    model = tf.keras.Sequential()
    
    # Tuning the memory units (16 to 64)
    hp_units = hp.Int('units', min_value=16, max_value=64, step=16)
    model.add(tf.keras.layers.LSTM(units=hp_units, input_shape=(60, 5), return_sequences=False))
    
    # Tuning the dropout (0.1 to 0.4)
    hp_dropout = hp.Float('dropout', 0.1, 0.4, step=0.1)
    model.add(tf.keras.layers.Dropout(hp_dropout))
    
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # 0.0 to 1.0 Risk Score
    
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- STEP 3: SIMULATED FEDERATED LEARNING ---
def train_federated(X_train, y_train, best_hps):
    print("\n🌐 Starting Simulated Federated Learning (3 Clients)...")
    
    # Split training data into 3 "Client Phones"
    X_clients = np.array_split(X_train, 3)
    y_clients = np.array_split(y_train, 3)
    
    # Create the Global Model
    global_model = build_model(best_hps)
    
    for i in range(3):
        print(f"📱 Training on Client {i+1}/3...")
        # Client trains on their local data
        global_model.fit(X_clients[i], y_clients[i], epochs=5, batch_size=32, verbose=0)
        
        # In real FL, we would 'average' weights here. 
        # Since this is a simulation, the model state is updated sequentially.
        
    return global_model

# # --- STEP 4: OFFLINE SHAP VALIDATION (The Reason Code Engine) ---
# def validate_with_shap(model, X_train, context_train):
#     print("\n🔍 Performing Offline SHAP Validation for Reason Codes...")
    
#     # We use a small background sample to explain the model's behavior
#     background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
#     explainer = shap.GradientExplainer(model, background)
    
#     # Calculate SHAP values for the vitals (The 5 in the window)
#     shap_values = explainer.shap_values(background)
    
#     # Calculate Global Feature Importance
#     vitals_importance = np.abs(shap_values[0]).mean(axis=(0, 1))
#     vitals_list = ['HR', 'HRV', 'SPO2', 'BP_S', 'BP_D']
    
#     # Store these weights. The App uses these to decide the 'Reason'.
#     xai_weights = dict(zip(vitals_list, vitals_importance))
    
#     # Add manual context weights (Age/Sleep/Activity) for the RAG engine
#     xai_weights['AGE'] = 0.15
#     xai_weights['ACTIVITY'] = 0.10
#     xai_weights['SLEEP'] = 0.20
    
#     joblib.dump(xai_weights, 'shield_xai_weights.pkl')
#     print("✅ Reason Code weights saved to 'shield_xai_weights.pkl'")
#     return xai_weights

# --- STEP 4: REFIXED for 5 Vitals ---
def validate_with_shap(model, X_train, context_train):
    print("\n🔍 Performing SHAP Validation...")
    
    # 1. Select background and explain
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    explainer = shap.GradientExplainer(model, background)
    
    # 2. Get SHAP values
    # shape will be (samples, 60, 5)
    shap_values = explainer.shap_values(background)
    
    # Handle both list and array outputs from SHAP (sometimes it returns a list)
    if isinstance(shap_values, list):
        shap_data = shap_values[0]
    else:
        shap_data = shap_values

    # 3. Calculate importance for each of the 5 features
    # We mean across samples (axis 0) and across time steps (axis 1)
    # This results in exactly 5 values
    vitals_importance = np.abs(shap_data).mean(axis=(0, 1))
    
    vitals_list = ['HR', 'HRV', 'SPO2', 'BP_S', 'BP_D']
    
    # 4. Create clean dictionary with standard floats
    xai_weights = {}
    for i in range(len(vitals_list)):
        xai_weights[vitals_list[i]] = float(vitals_importance[i])
    
    # 5. Save to JSON
    with open('shield_weights.json', 'w') as f:
        json.dump(xai_weights, f)
        
    print("✅ Successfully saved 5 vitals to shield_weights.json")
    print("Weights found:", xai_weights)
    
    return xai_weights

# --- EXECUTION ---
X, y, context = load_and_impute()
X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, context, test_size=0.2)

# A. Tune
tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=5, directory='tuning')
tuner.search(X_train, y_train, epochs=5, validation_split=0.1, verbose=0)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# B. Federated Train
final_model = train_federated(X_train, y_train, best_hps)

# C. Validate & Generate Reason Logic
weights = validate_with_shap(final_model, X_train, c_train)

# D. Export to TFLite
final_model.save('shield_v2.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# with open('shield_v2.tflite', 'wb') as f:
#     f.write(tflite_model)

# converter = tf.lite.TFLiteConverter.from_keras_model(final_model)

run_model = tf.function(lambda x: final_model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 60, 5], final_model.inputs[0].dtype)
)

# 2. Convert from the concrete function instead of the Keras model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# 2. FORCE Built-ins only (This removes the "unresolved-ops" error)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# 3. Handle LSTM/RNN conversion if the model uses them
# (This lowers complex ops to basic ones that mobile supports)
converter._experimental_lower_tensor_list_ops = True

try:
    tflite_model = converter.convert()
    with open('shield_v2.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Success: Model exported with TFLITE_BUILTINS only.")
except Exception as e:
    print("Error: Model uses operations not supported by standard TFLite.")
    print("Consider replacing complex layers (like LSTMs) with simpler ones or check activations.")
    print(e)

print("\n🚀 S.H.I.E.L.D. Deployment Package Ready!")