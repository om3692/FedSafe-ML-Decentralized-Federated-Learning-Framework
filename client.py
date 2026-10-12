# ==========================================
# Project: FedSafe-ML: Decentralized Federated Learning Framework
# Author: Om Awalekar
# PRN: 32458020002
# File: client.py
# Description: The local decentralized node running CIFAR-10 architecture.
# ==========================================

import argparse
import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
import logging
import sys

from model import CifarCNN
from dataset import load_partition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - CLIENT: %(message)s')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FedSafeClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        epochs = config.get("epochs", 1) 
        
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / len(self.testloader.dataset)
        avg_loss = loss / len(self.testloader)
        
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser(description="FedSafe-ML Client Node")
    parser.add_argument("--client-id", type=int, required=True, help="Unique ID for this client")
    parser.add_argument("--num-clients", type=int, default=3, help="Total number of clients")
    args = parser.parse_args()

    logging.info(f"Local training will utilize device: {DEVICE}")
    logging.info(f"Initializing Client {args.client_id}...")
    
    try:
        trainloader, testloader = load_partition(client_id=args.client_id, num_clients=args.num_clients)
        model = CifarCNN().to(DEVICE)
        client = FedSafeClient(model, trainloader, testloader)
        
        logging.info(f"Client {args.client_id} attempting to connect to orchestration server...")
        fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
        
    except ConnectionError:
        logging.error("Failed to connect to the central server. Ensure server.py is running.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal client error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()