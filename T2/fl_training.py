import flwr as fl
from flwr.simulation import start_simulation

# Define client factory
def client_fn(cid: str):
    return HeartDiseaseClient(cid)

# Run FL simulation (10 simulated clients)
start_simulation(
    client_fn=client_fn,
    num_clients=10,
    client_resources={"num_cpus": 1, "num_gpus": 0},
    config=fl.server.ServerConfig(num_rounds=3)
)
