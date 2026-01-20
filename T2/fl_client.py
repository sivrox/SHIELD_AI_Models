import flwr as fl
import numpy as np
from sklearn.metrics import accuracy_score

# Load your data partitions (Flower Datasets automates this)
def load_partition(cid):  # Client ID
    # Simulate client data (replace with UCI partition)
    X_client, y_client = ...  # Your client-specific data
    return (X_client, y_client)

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = create_model(13)  # UCI has 13 features
        self.X, self.y = load_partition(cid)

    def get_parameters(self, config):
        return [val.numpy() for val in self.model.trainable_variables]

    def fit(self, parameters, config):
        # Load global model params
        for i, val in enumerate(self.model.trainable_variables):
            val.assign(parameters[i])
        
        # Train locally (1 epoch for demo)
        self.model.fit(self.X, self.y, epochs=1, verbose=0)
        
        # Return updated params + stats
        return [val.numpy() for val in self.model.trainable_variables], len(self.X), {}

    def evaluate(self, parameters, config):
        # Load params, evaluate
        for i, val in enumerate(self.model.trainable_variables):
            val.assign(parameters[i])
        loss, acc = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {"accuracy": float(acc)}
