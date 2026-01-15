import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # Used to save our 'Equalizer' for the app

# --- STEP 1: LOADING & ALIGNING (The Recruitment) ---
def load_data():
    # Load both our 'History Book' and our 'Simulation'
    public_df = pd.read_csv('public_vitals.csv')
    synthetic_df = pd.read_csv('synthetic_vitals.csv')
    
    # We ensure they both follow the exact same column order
    cols = ['age', 'hr', 'bp_s', 'hrv', 'bp_d', 'spo2', 'activity', 'target']
    df = pd.concat([public_df[cols], synthetic_df[cols]], ignore_index=True)
    
    # Separate the Questions (X) from the Answer (y)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # --- STEP 2: DATA SPLITTING (Study vs. Final Exam) ---
    # We split into 3 groups:
    # 1. Training (70%): The AI studies these cases repeatedly.
    # 2. Validation (15%): The AI takes 'Pop Quizzes' during study.
    # 3. Test (15%): The 'Final Exam' data the AI NEVER sees during study.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- STEP 3: PREPROCESSING (The Equalizer) ---
def preprocess_data(X_train, X_val, X_test):
    # Why: BP is ~140, but SpO2 is ~0.95. AI thinks 140 is 100x more important.
    # StandardScaler makes the 'average' 0 and the 'spread' 1 for every vital.
    scaler = StandardScaler()
    
    # We 'Fit' only on the training data (to learn the scales)
    # Then we 'Transform' all three sets.
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # We save the scaler! The React Native app needs this to 'equalize' 
    # the real user's data before sending it to the brain.
    joblib.dump(scaler, 'shield_scaler.pkl')
    
    return X_train_scaled, X_val_scaled, X_test_scaled

# --- STEP 4: HYPERPARAMETER TUNING (Finding the Magic Settings) ---
def build_tunable_model(hp):
    model = tf.keras.Sequential()
    
    # Knob 1: How many neurons in the first layer?
    # We test 16, 32, 48, or 64. 
    # Too many = AI 'memorizes' (Overfitting). Too few = AI is 'slow' (Underfitting).
    hp_units = hp.Int('units', min_value=16, max_value=64, step=16)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu', input_shape=(7,)))
    
    # Knob 2: Dropout (The 'Anti-Lazy' layer)
    # We randomly turn off 10% to 30% of neurons so others work harder.
    hp_dropout = hp.Float('dropout', 0.1, 0.3, step=0.1)
    model.add(tf.keras.layers.Dropout(hp_dropout))
    
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Knob 3: Learning Rate (How fast the coach talks)
    # 1e-2 is fast (0.01), 1e-4 is slow (0.0001).
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- EXECUTION ---
X_train, X_val, X_test, y_train, y_val, y_test = load_data()
X_train_s, X_val_s, X_test_s = preprocess_data(X_train, X_val, X_test)

# Start the 'Tuner' search
tuner = kt.Hyperband(
    build_tunable_model,
    objective='val_accuracy',
    max_epochs=20,
    directory='tuning_lab',
    project_name='shield_v1'
)

print("\n--- PHASE 1: TUNING (Searching for Best Settings) ---")
tuner.search(X_train_s, y_train, epochs=20, validation_data=(X_val_s, y_val))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# --- STEP 5: FINAL TRAINING ---
print("\n--- PHASE 2: FINAL TRAINING (Building the Brain) ---")
# We build the model using the 'Best Knobs' we found in tuning
model = tuner.hypermodel.build(best_hps)

# 'EarlyStopping' stops the training if the AI stops getting smarter.
# This prevents wasting battery and overfitting.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    X_train_s, y_train, 
    epochs=50, 
    validation_data=(X_val_s, y_val),
    callbacks=[stop_early]
)

# --- STEP 6: EVALUATION (The Final Exam) ---
print("\n--- PHASE 3: THE FINAL EXAM ---")
test_loss, test_acc = model.evaluate(X_test_s, y_test)
print(f"Final Model Accuracy on Unseen Data: {test_acc*100:.2f}%")

# Save for Mobile
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('shield_v1.tflite', 'wb') as f:
    f.write(tflite_model)