import pandas as pd
import numpy as np

def generate_synthetic_data(samples=10000):
    np.random.seed(42)
    
    # 1. Random Vitals
    age = np.random.randint(18, 95, samples)
    hr = np.random.randint(40, 160, samples)
    bp_s = np.random.randint(80, 200, samples)
    spo2 = np.random.randint(85, 100, samples)
    activity = np.random.randint(0, 3, samples) # 0:Rest, 1:Walk, 2:Run
    
    # 2. Logic-Based Generation
    target = []
    hrv = []
    
    for i in range(samples):
        # We start with the assumption they are healthy
        is_risk = 0
        
        # RULE: High BP + Low Activity = High Risk
        if bp_s[i] > 150 and activity[i] == 0: is_risk = 1
        
        # RULE: Low Oxygen = Emergency
        if spo2[i] < 92: is_risk = 1
        
        # RULE: Age-Based Risk
        if age[i] > 75 and bp_s[i] > 140: is_risk = 1
        
        target.append(is_risk)
        
        # HRV Generation based on risk (Healthy = High HRV, Sick = Low HRV)
        if is_risk == 1:
            hrv.append(np.random.normal(30, 10)) # Low & Stressed
        else:
            hrv.append(np.random.normal(70, 15)) # High & Healthy

    # 3. Create Table
    df = pd.DataFrame({
        'age': age, 'hr': hr, 'bp_s': bp_s, 'hrv': hrv,
        'bp_d': bp_s * 0.7, 'spo2': spo2, 'activity': activity, 'target': target
    })
    
    # Clean up numbers
    df['hrv'] = df['hrv'].clip(10, 120).round(1)
    df.to_csv('synthetic_vitals.csv', index=False)
    print("Synthetic Dataset saved: synthetic_vitals.csv")

if __name__ == "__main__":
    generate_synthetic_data()