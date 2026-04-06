"""
S.H.I.E.L.D. Federated Learning Server
=======================================
FastAPI server that implements the FedAvg aggregation protocol.

What it does:
  1. Hosts the current global LSTM model
  2. Accepts weight updates from FL clients (hospitals/devices)
  3. Aggregates updates using Federated Averaging (FedAvg)
  4. Serves the updated global model back to clients
  5. Exports updated TFLite after each aggregation round

How it works:
  - Clients POST their locally-trained model weights to /fl/submit
  - When enough clients have submitted (min_clients), server runs FedAvg
  - Server averages all client weights proportional to their dataset size
  - Updated global model is available via GET /fl/global-model
  - After aggregation, new TFLite is exported to outputs/

Requirements: pip install fastapi uvicorn tensorflow numpy
Usage: python fl_server.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

#Configuration
model_dir = 'outputs'
global_model_path = os.path.join(model_dir, 'shield_global.keras')
tflite_path = os.path.join(model_dir, 'shield.tflite')
scaler_path = os.path.join(model_dir, 'scaler_stats.json')
weights_path = os.path.join(model_dir, 'shield_weights.json')
min_clients = 2       # minimum clients before aggregation triggers
window_size = 60
n_features = 8

os.makedirs(model_dir, exist_ok=True)

app = FastAPI(title="S.H.I.E.L.D FL Server", version="1.0")


#Build Model

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_input_shape=(1, window_size, n_features)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, unroll=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False, unroll=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# === Global State ===

class FLState:
    def __init__(self):
        self.round = 0
        self.pending_updates = []    # list of (weights, n_samples, client_id)
        self.history = []            # aggregation history
        self.global_model = None
        self.load_or_init_model()

    def load_or_init_model(self):
        """Load existing global model or initialize a new one."""
        self.global_model = build_model()
        if os.path.exists(global_model_path):
            self.global_model.load_weights(global_model_path)
            print(f"Loaded existing global model from {global_model_path}")
        else:
            print("Initialized new global model")

    def add_update(self, weights, n_samples, client_id):
        """Store a client's weight update."""
        self.pending_updates.append({
            'weights': weights,
            'n_samples': n_samples,
            'client_id': client_id,
            'timestamp': datetime.now().isoformat()
        })
        print(f"Received update from {client_id} ({n_samples} samples)")

    def aggregate(self):
        """Run FedAvg: weighted average of all pending client weights."""
        if len(self.pending_updates) < min_clients:
            return False

        self.round += 1
        print(f"\n--- FL Round {self.round}: Aggregating {len(self.pending_updates)} clients ---")

        # Weighted average (clients with more data have more influence)
        total_samples = sum(u['n_samples'] for u in self.pending_updates)
        global_weights = self.global_model.get_weights()
        new_weights = [np.zeros_like(w) for w in global_weights]

        for update in self.pending_updates:
            weight_factor = update['n_samples'] / total_samples
            for i, layer_w in enumerate(update['weights']):
                new_weights[i] += layer_w * weight_factor

        # Update global model
        self.global_model.set_weights(new_weights)
        self.global_model.save(global_model_path)

        # Export updated TFLite
        self.export_tflite()

        # Record history
        self.history.append({
            'round': self.round,
            'n_clients': len(self.pending_updates),
            'total_samples': total_samples,
            'clients': [u['client_id'] for u in self.pending_updates],
            'timestamp': datetime.now().isoformat()
        })

        # Clear pending updates for next round
        self.pending_updates = []
        print(f"--- Round {self.round} complete. Global model updated. ---\n")
        return True

    def export_tflite(self):
        """Convert current global model to TFLite format."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.global_model)
        tflite_bytes = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_bytes)
        print(f"Exported TFLite: {len(tflite_bytes)/1024:.1f} KB")

    def get_weights_as_list(self):
        """Return global model weights as JSON-serializable lists."""
        return [w.tolist() for w in self.global_model.get_weights()]


fl_state = FLState()


# === Request/Response Models ===

class WeightUpdate(BaseModel):
    client_id: str
    n_samples: int
    weights: List  # nested list of layer weights

class AggregationResponse(BaseModel):
    success: bool
    round: int
    n_clients: int
    message: str


# === API Endpoints ===

@app.get("/")
def root():
    return {
        "service": "S.H.I.E.L.D. FL Server",
        "current_round": fl_state.round,
        "pending_updates": len(fl_state.pending_updates),
        "min_clients_for_aggregation": min_clients
    }

@app.get("/fl/status")
def fl_status():
    """Get current FL server status."""
    return {
        "round": fl_state.round,
        "pending_updates": len(fl_state.pending_updates),
        "min_clients": min_clients,
        "history": fl_state.history[-5:]  # last 5 rounds
    }

@app.post("/fl/submit")
def submit_weights(update: WeightUpdate):
    """Client submits locally-trained weights."""
    try:
        # Convert nested lists back to numpy arrays
        weights = [np.array(w, dtype=np.float32) for w in update.weights]

        # Validate weight shapes match global model
        global_shapes = [w.shape for w in fl_state.global_model.get_weights()]
        client_shapes = [w.shape for w in weights]
        if global_shapes != client_shapes:
            raise HTTPException(400, f"Weight shape mismatch. "
                                     f"Expected {global_shapes}, got {client_shapes}")

        fl_state.add_update(weights, update.n_samples, update.client_id)

        # Auto-aggregate if enough clients
        if len(fl_state.pending_updates) >= min_clients:
            fl_state.aggregate()
            return {"status": "aggregated", "round": fl_state.round,
                    "message": f"Round {fl_state.round} complete. Model updated."}

        remaining = min_clients - len(fl_state.pending_updates)
        return {"status": "pending", "round": fl_state.round,
                "message": f"Update received. Waiting for {remaining} more client(s)."}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing update: {str(e)}")

@app.post("/fl/aggregate")
def force_aggregate():
    """Manually trigger aggregation (for demo purposes)."""
    if len(fl_state.pending_updates) == 0:
        raise HTTPException(400, "No pending updates to aggregate.")
    success = fl_state.aggregate()
    if success:
        return {"status": "aggregated", "round": fl_state.round}
    return {"status": "failed", "message": f"Need at least {min_clients} clients."}

@app.get("/fl/global-model")
def get_global_model():
    """Download the current global TFLite model."""
    if not os.path.exists(tflite_path):
        fl_state.export_tflite()
    return FileResponse(tflite_path, filename="shield.tflite",
                        media_type="application/octet-stream")

@app.get("/fl/global-weights")
def get_global_weights():
    """Get global model weights as JSON (for clients that train with TF)."""
    return {"round": fl_state.round,
            "weights": fl_state.get_weights_as_list()}

@app.get("/fl/history")
def get_history():
    """Get full FL training history."""
    return {"total_rounds": fl_state.round, "history": fl_state.history}


# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    print("S.H.I.E.L.D FL Aggregator Server\n")
    print(f"  Min clients for aggregation: {min_clients}")
    print(f"  Model output: {tflite_path}")
    print(f"  Starting on http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)