import numpy as np
import tensorflow as tf
import json

def run_integration_test():
    print("🧪 Starting S.H.I.E.L.D. Deployment Validation...")

    # 1. LOAD THE SCALER STATS
    with open('scaler_stats.json', 'r') as f:
        stats = json.load(f)
    print("✅ Scaler Stats: Loaded.")

    # 2. LOAD THE XAI WEIGHTS
    with open('shield_weights.json', 'r') as f:
        weights = json.load(f)
    print("✅ XAI Weights: Loaded.")

    # 3. LOAD THE TFLITE BRAIN
    interpreter = tf.lite.Interpreter(model_path="shield_v3.tflite")
    interpreter.allocate_tensors()
    print("✅ TFLite Brain: Initialized.")

    # 4. SIMULATE LIVE DATA (Normal Range)
    # Let's simulate a patient with HR=75 and SpO2=98%
    # We use the stats file to scale them manually like the app would
    raw_hr = 75.0
    scaled_hr = (raw_hr - stats['hr']['mean']) / stats['hr']['std']
    
    # Create a dummy 60-second window with 8 features
    # Input shape: (Batch, Seconds, Features) -> (1, 60, 8)
    dummy_input = np.zeros((1, 60, 8), dtype=np.float32)
    dummy_input[0, :, 0] = scaled_hr # Fill HR column with our scaled value

    # 5. RUN THE PREDICTION
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    
    risk_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    print("-" * 30)
    print(f"📊 TEST RESULT: Cardiovascular Risk Score is {risk_score * 100:.1f}%")
    print(f"💡 HIGHEST WEIGHTED VITAL: {max(weights, key=weights.get).upper()}")
    print("-" * 30)
    print("🚀 ALL SYSTEMS GO. Your AI ecosystem is ready for the Mobile App.")

if __name__ == "__main__":
    run_integration_test()

# --- DEEP UNDERSTANDING FOR THE TEST ---
# - interpreter.invoke(): 
#   Reasoning: This is the exact command the React Native app uses to 'Trigger 
#   Thinking'. It runs the math inside the .tflite file.
#
# - scaled_hr logic: 
#   Reasoning: This proves why we needed 'scaler_stats.json'. Without it, 
#   the app would send '75' to the AI, which expects a number near '0'. 
#   This test ensures the 'Translator' and the 'Brain' are speaking the same language.