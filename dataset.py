# ==========================================
# Project: FedSafe-ML: Decentralized Federated Learning Framework
# Author: Om Awalekar
# PRN: 32458020002
# File: dataset.py
# Description: Handles secure, siloed data partitioning for CIFAR-10.
# ==========================================

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def load_partition(client_id: int, num_clients: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    if client_id >= num_clients or client_id < 0:
        raise ValueError(f"Invalid client_id {client_id}. Must be between 0 and {num_clients-1}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        # download=True will safely skip downloading since the file is already in ./data
        full_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        full_test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    except Exception as e:
        raise RuntimeError(f"Failed to load CIFAR-10 dataset: {e}")

    train_partition_size = len(full_train_dataset) // num_clients
    test_partition_size = len(full_test_dataset) // num_clients

    train_lengths = [train_partition_size] * num_clients
    test_lengths = [test_partition_size] * num_clients

    train_lengths[-1] += len(full_train_dataset) - sum(train_lengths)
    test_lengths[-1] += len(full_test_dataset) - sum(test_lengths)

    generator = torch.Generator().manual_seed(42)
    
    train_datasets = random_split(full_train_dataset, train_lengths, generator=generator)
    test_datasets = random_split(full_test_dataset, test_lengths, generator=generator)

    client_train_data = train_datasets[client_id]
    client_test_data = test_datasets[client_id]

    trainloader = DataLoader(client_train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(client_test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader