"""
S.H.I.E.L.D. FL Demo Runner
============================
Automates the FL demonstration:
  1. Starts the FL server in the background
  2. Runs 3 FL clients (simulating 3 hospitals)
  3. Shows the aggregation results
  4. Verifies the updated model

Usage: python fl_demo.py
Make sure shield_dataset.csv and outputs/ folder exist first.
Requirements: pip install tensorflow numpy requests pandas fastapi uvicorn
"""

import subprocess
import time
import sys
import os
import requests
import json

server_url = "http://localhost:8080"
dataset = "shield_training_dataset.csv"

def wait_for_server(timeout=30):
    print("Waiting for FL server to start", end="")
    for _ in range(timeout):
        try:
            resp = requests.get(f"{server_url}/", timeout=2)
            if resp.status_code == 200:
                print(" Ready!")
                return True
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print("TIMEOUT")
    return False

#Run a single FL client and wait for it to finish.
def run_client(client_id, hospital_id):
    print(f"Running FL Client: {client_id} (Hospital {hospital_id})\n")
    result = subprocess.run([
        sys.executable, "fl_client.py",
        "--client-id", client_id,
        "--data", dataset,
        "--hospital", str(hospital_id)
    ], capture_output=False)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("S.H.I.E.L.D Federated Learning Demo")
    print("=" * 60)

    # Check dataset exists
    if not os.path.exists(dataset):
        print(f"ERROR: {dataset} not found. Run shield_dataset_generator.py first.")
        sys.exit(1)

    # Start FL server in background
    print("\n[1/4] Starting FL server...")
    server_proc = subprocess.Popen(
        [sys.executable, "fl_server.py"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not wait_for_server():
        print("ERROR: FL server failed to start")
        server_proc.kill()
        sys.exit(1)

    # Check server status
    status = requests.get(f"{SERVER_URL}/").json()
    print(f"  Server running — Round: {status['current_round']}")

    # Run 3 hospital clients sequentially
    print("\n[2/4] Running FL clients (3 hospitals)...")
    hospitals = [
        ("hospital_0", 0),
        ("hospital_1", 1),
        ("hospital_2", 2),
    ]

    for client_id, hospital_id in hospitals:
        success = run_client(client_id, hospital_id)
        if not success:
            print(f"ERROR: Client {client_id} failed")
        time.sleep(2)  # small pause between clients

    # Check final status
    print(f"\n[3/4] Checking FL results...")
    status = requests.get(f"{server_url}/fl/status").json()
    print(f"Completed rounds: {status['round']}")
    print(f"Pending updates: {status['pending_updates']}")
    if status['history']:
        last = status['history'][-1]
        print(f"Last round: {last['n_clients']} clients, "
              f"{last['total_samples']:,} total samples")

    # Verify output files exist
    print(f"\n[4/4] Verifying outputs...")
    for f in ['shield.tflite', 'scaler_stats.json', 'shield_weights.json']:
        path = os.path.join('outputs', f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"{f}: {size/1024:.1f} KB")
        else:
            print(f"{f}: MISSING")

    # Cleanup
    print("FL Demo Complete:")
    print("Model saved in outputs/shield.tflite")

    # Keep server running for inspection
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        server_proc.kill()
        print("\nServer stopped.")

if __name__ == "__main__":
    main()