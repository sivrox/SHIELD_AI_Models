import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
import os

# --- STEP 1: LOAD TEST DATA ---
def load_test_set():
    print("📥 Loading windows and labels for evaluation...")
    # Loading the binaries created by your combine_and_convert.py
    X = np.load('X_windows.npy')
    y = np.load('y_labels.npy')
    
    # We use the same random_state (42) as the trainer 
    # to ensure we are testing on the 'Final Exam' data, not the 'Study' data.
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute any flickers (NaNs) just in case
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    return X_test, y_test

# --- STEP 2: RUN PREDICTIONS ---
def get_model_predictions(X_test):
    print("🧠 Loading model and generating predictions...")
    
    # Note: This assumes you saved your model as .h5 in the trainer.
    # If you only have the .tflite, we would use the TFLite Interpreter.
    if os.path.exists('shield_v2.h5'):
        model = tf.keras.models.load_model('shield_v2.h5')
        # Predictions will be probabilities (e.g., 0.85)
        y_probs = model.predict(X_test).ravel()
        # Classes will be hard 0 or 1 based on 50% threshold
        y_pred = (y_probs > 0.5).astype(int)
        return y_probs, y_pred
    else:
        print("❌ Error: 'shield_v2.h5' not found. Please save your model in the trainer first.")
        return None, None

# --- STEP 3: VISUAL ANALYSIS ---
def run_full_evaluation(y_test, y_probs, y_pred):
    print("\n📊 Generating Medical Performance Reports...")

    # A. THE CONFUSION MATRIX (The 'Logic Check')
    # This shows exactly where the AI got confused.
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Healthy', 'Risk'], 
                yticklabels=['Healthy', 'Risk'])
    plt.title('Decision Quality: Actual vs Predicted')
    plt.ylabel('Clinical Truth')
    plt.xlabel('AI Prediction')

    # B. THE ROC CURVE (The 'Grade')
    # Shows the trade-off between catching risks and avoiding false alarms.
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve: Reliability Analysis')
    plt.xlabel('False Alarms (FPR)')
    plt.ylabel('Caught Risks (TPR)')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig('clinical_performance_report.png')
    print("✅ Performance charts saved to 'clinical_performance_report.png'")

    # C. TEXT-BASED REPORT (Precision/Recall)
    print("\n--- DETAILED MEDICAL METRICS ---")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'At-Risk']))
    print(f"Overall Model Grade (AUC): {roc_auc:.4f}")

# --- EXECUTION ---
if __name__ == "__main__":
    X_test, y_test = load_test_set()
    y_probs, y_pred = get_model_predictions(X_test)
    
    if y_probs is not None:
        run_full_evaluation(y_test, y_probs, y_pred)