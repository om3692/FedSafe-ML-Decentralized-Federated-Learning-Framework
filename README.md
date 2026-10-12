# FedSafe-ML: Decentralized Federated Learning Framework


## Overview
FedSafe-ML is a production-grade simulation of a privacy-preserving Federated Learning (FL) pipeline. It demonstrates the capability to train deep neural networks on decentralized data silos without exchanging raw data, mitigating data privacy risks while still converging on a highly accurate global model.

## System Architecture

The simulation utilizes the **Flower (flwr)** framework to orchestrate communication between the centralized server and localized client edge-devices.

```mermaid
graph TD
    subgraph Centralized Aggregation Layer
    S[Aggregation Server\nFedAvg Strategy]
    end
    
    subgraph Decentralized Edge Devices
    C1[Client 0\nLocal Data: Shard A]
    C2[Client 1\nLocal Data: Shard B]
    C3[Client 2\nLocal Data: Shard C]
    end

    S -- Broadcasts Global Model (Weights) --> C1
    S -- Broadcasts Global Model (Weights) --> C2
    S -- Broadcasts Global Model (Weights) --> C3
    
    C1 -. Trains on Shard A & Returns Gradients .-> S
    C2 -. Trains on Shard B & Returns Gradients .-> S
    C3 -. Trains on Shard C & Returns Gradients .-> S
    
    classDef server fill:#f9f,stroke:#333,stroke-width:4px;
    classDef client fill:#bbf,stroke:#333,stroke-width:2px;
    class S server;
    class C1,C2,C3 client;
