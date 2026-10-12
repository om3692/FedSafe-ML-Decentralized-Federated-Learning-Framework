#!/bin/bash
# ==========================================
# Project: FedSafe-ML: Decentralized Federated Learning Framework
# Author: Om Awalekar
# PRN: 32458020002
# File: run.sh
# Description: Automated execution script to orchestrate the simulation.
# ==========================================

echo "=========================================="
echo " Starting FedSafe-ML Simulation"
echo " Author: Om Awalekar | PRN: 32458020002"
echo "=========================================="

# Setup signal trapping to gracefully kill all background processes on Ctrl+C
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo "[INFO] Starting centralized aggregation server..."
python server.py &
SERVER_PID=$!

# Wait for the server to bind to the port
sleep 3

echo "[INFO] Spawning decentralized client nodes..."
python client.py --client-id 0 --num-clients 3 &
python client.py --client-id 1 --num-clients 3 &
python client.py --client-id 2 --num-clients 3 &

echo "[INFO] Simulation running in background. Aggregation will complete over 5 rounds."
echo "[INFO] Press Ctrl+C to terminate early."

# Wait for the server process to naturally finish
wait $SERVER_PID

echo "[INFO] FedSafe-ML simulation completed successfully."