# ==========================================
# Project: FedSafe-ML: Decentralized Federated Learning Framework
# Author: Om Awalekar
# PRN: 32458020002
# File: run.ps1
# Description: Windows execution script to orchestrate the FL simulation.
# ==========================================

Write-Host "=========================================="
Write-Host " Starting FedSafe-ML Simulation (Windows)"
Write-Host " Author: Om Awalekar | PRN: 32458020002"
Write-Host "=========================================="

$ScriptDir = $PWD.Path

Write-Host "[INFO] Starting centralized aggregation server in a new window..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir'; .\fedsafe_env\Scripts\activate; python server.py"

# Wait for the server to initialize on port 8080
Start-Sleep -Seconds 4

Write-Host "[INFO] Spawning 3 decentralized client nodes in new windows..."
for ($i=0; $i -lt 3; $i++) {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir'; .\fedsafe_env\Scripts\activate; python client.py --client-id $i --num-clients 3"
}

Write-Host "[INFO] All nodes started successfully!"
Write-Host "[INFO] Please check the 4 newly opened terminal windows to observe the federated training process."