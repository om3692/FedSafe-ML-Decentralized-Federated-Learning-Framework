# ==========================================
# Project: FedSafe-ML: Decentralized Federated Learning Framework
# Author: Om Awalekar
# PRN: 32458020002
# File: server.py
# Description: The centralized orchestrator. Aggregates model updates 
#              using the Federated Averaging (FedAvg) algorithm without 
#              ever accessing raw client data.
# ==========================================

import flwr as fl
import logging
from typing import Dict, Optional, Tuple
import sys

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - SERVER: %(message)s')

def get_evaluate_fn():
    """
    Return an evaluation function for centralized server-side evaluation.
    (Optional in pure FL, but good for tracking global convergence).
    We skip it here to strictly rely on client-side federated evaluation,
    which is truer to strict privacy environments.
    """
    return None

def main():
    """
    Initializes and starts the Flower server with defined aggregation strategies.
    """
    logging.info("Initializing FedAvg Orchestration Strategy...")
    
    # Configure the Federated Averaging strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,               # Sample 100% of available clients for training
        fraction_evaluate=1.0,          # Sample 100% of available clients for evaluation
        min_fit_clients=3,              # Wait until 3 clients are available for training
        min_evaluate_clients=3,         # Wait until 3 clients are available for evaluation
        min_available_clients=3,        # Total minimum clients required to start the simulation
        evaluate_fn=get_evaluate_fn(),
    )

    logging.info("Starting Federated Server on [::]:8080...")
    
    try:
        # Start the server synchronously
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=5), # 5 communication rounds
            strategy=strategy,
        )
        logging.info("Federated Learning simulation completed successfully.")
    except OSError as e:
        logging.error(f"Network bind error (Port 8080 might be in use): {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected server crash: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()