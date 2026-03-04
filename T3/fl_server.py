import flwr as fl
import tensorflow as tf
import numpy as np
import json
import shap
from lstm_model import get_shield_model

# Initialize the global master model
master_model = get_shield_model()
final_scaler_stats = {}

def finalize_deployment_files(weights):
    """
    The Assembly Line: Generates TFLite, SHAP Weights, and Scaler Stats.
    """
    print("\n🏁 Federated Rounds Complete. Finalizing all 3 deployment files...")
    master_model.set_weights(weights)

    # 1. SAVE SCALER STATS
    if final_scaler_stats:
        with open('scaler_stats.json', 'w') as f:
            json.dump(final_scaler_stats, f, indent=2)
        print("✅ 1/3: scaler_stats.json saved.")

    # 2. SAVE SHAP WEIGHTS
    print("🔍 2/3: Running SHAP Analysis...")
    background = np.random.normal(size=(50, 60, 8)).astype(np.float32)
    explainer = shap.GradientExplainer(master_model, background)
    shap_v = explainer.shap_values(background[:10])
    mean_impact = np.abs(shap_v[0]).mean(axis=(0, 1))
    
    features = ['hr', 'hrv', 'spo2', 'bp_s', 'bp_d', 'age', 'activity', 'sleep']
    weights_dict = {features[i]: float(mean_impact[i]) for i in range(8)}
    with open('shield_weights.json', 'w') as f:
        json.dump(weights_dict, f, indent=2)

    # 3. SAVE TFLITE MODEL
    print("📦 3/3: Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(master_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open('shield_v4.tflite', 'wb') as f:
        f.write(tflite_model)
    print("\n🚀 MISSION COMPLETE: All 3 files are ready for Frontend.")

class ShieldStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        """
        THE FIX IS HERE: results[0][1].metrics
        We access index [1] to get the FitRes object inside the tuple.
        """
        if server_round == 1 and results:
            global final_scaler_stats
            # result[0] is (ClientProxy, FitRes). We need index [1] for metrics.
            try:
                stats_str = results[0][1].metrics.get("scaler_json", "{}")
                final_scaler_stats = json.loads(stats_str)
                print("📊 Server: Scaler stats successfully captured from Client.")
            except Exception as e:
                print(f"⚠️ Warning: Could not parse scaler stats: {e}")

        agg_weights = super().aggregate_fit(server_round, results, failures)
        
        # We run 5 rounds of global learning
        if server_round == 5 and agg_weights is not None:
            weights_np = fl.common.parameters_to_ndarrays(agg_weights)
            finalize_deployment_files(weights_np)
        return agg_weights

if __name__ == "__main__":
    print("🛡️ S.H.I.E.L.D. Aggregator online. Listening on Port 8080...")
    
    strategy = ShieldStrategy(
        min_fit_clients=1, # Works even if you only test with 1 phone
        min_available_clients=1,
        initial_parameters=fl.common.ndarrays_to_parameters(master_model.get_weights())
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )