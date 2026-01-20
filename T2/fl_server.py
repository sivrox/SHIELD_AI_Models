import flwr as fl

# Start FL server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),  # 3 training rounds
    strategy=fl.server.strategy.FedAvg()  # Simple averaging
)
