# Federated Learning — Algorithm Comparison

  Implements and compares four federated learning algorithms on MNIST across simulated distributed clients.

  ## Algorithms

  | Algorithm | Description |
  |-----------|-------------|
  | FedAvg | Standard federated averaging |
  | FedProx | FedAvg + proximal term (μ=0.01) for heterogeneous data |
  | CyclicWeightTransfer | Sequential weight passing between clients |
  | DPFedAvg | FedAvg with differential privacy (ε=1.0) |

  ## Setup  

  ```bash
  pip install -r requirements.txt
  python main.py

  Config

  Default settings in main.py:
  - 5 clients, 10 rounds
  - Batch size: 32, LR: 0.01

  Results saved to results/ as JSON. Accuracy and timing plots generated automatically.

  Stack

  Python · PyTorch · scikit-learn · matplotlib
