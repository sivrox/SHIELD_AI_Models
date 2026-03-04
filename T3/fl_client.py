import flwr as fl
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import os
from lstm_model import get_shield_model

def get_deployment_stats():
    """Calculates scaling rules from the local CSV to send to the Boss."""
    if not os.path.exists('combined_training_data.csv'):
        return json.dumps({})
    
    data = pd.read_csv('combined_training_data.csv')
    features = ['hr', 'hrv', 'spo2', 'bp_s', 'bp_d', 'age', 'activity', 'sleep']
    
    stats = {}
    for col in features:
        if col == 'activity':
            stats['activity_level'] = {"mean": 0.0, "std": 1.0, "type": "categorical"}
        else:
            stats[col] = {
                "mean": float(data[col].mean()), 
                "std": float(data[col].std()),
                "type": "numerical"
            }
    return json.dumps(stats)

class ShieldClient(fl.client.NumPyClient):
    def __init__(self, model, X, y, stats_json):
        self.model = model
        self.X, self.y = X, y
        self.stats_json = stats_json

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        print("📱 Phone: Local study session in progress...")
        self.model.fit(self.X, self.y, epochs=3, batch_size=32, verbose=0)
        
        # We send back the learned weights and the scaler stats
        return self.model.get_weights(), len(self.X), {"scaler_json": self.stats_json}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        
        # CRITICAL: Always cast to standard Python float
        return float(loss), len(self.X), {"accuracy": float(acc)}

if __name__ == "__main__":
    # Load the high-speed binary data
    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
    
    model = get_shield_model()
    stats = get_deployment_stats()

    print("🛡️ Client connecting to S.H.I.E.L.D. Network...")
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=ShieldClient(model, X, y, stats)
    )