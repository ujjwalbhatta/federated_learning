from src.algorithms.base_algorithm import FederatedAlgorithm
import torch
import torch.nn as nn
from typing import List, Dict

class FedProx(FederatedAlgorithm):
    """FedProx implementation with proximal term"""
    def __init__(self, model: nn.Module, device: torch.device, mu: float = 0.01):
        self.model = model
        self.device = device
        self.mu = mu

    def proximal_term(self, local_params, global_params):
        """Calculate proximal term for FedProx"""
        proximal_term = 0
        for l_param, g_param in zip(local_params, global_params):
            proximal_term += (self.mu / 2) * torch.norm(l_param - g_param) ** 2
        return proximal_term

    def aggregate(self, client_states: List[Dict]) -> None:
        global_state = self.model.state_dict()
        for key in global_state.keys():
            client_weights = torch.stack([state['model_state'][key] for state in client_states])
            proximal_weights = client_weights + self.mu * (global_state[key].unsqueeze(0) - client_weights)
            global_state[key] = torch.mean(proximal_weights, dim=0)
        self.model.load_state_dict(global_state)
