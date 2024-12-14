import torch
import torch.nn as nn
from typing import List, Dict

class FedAvg:
    """FedAvg algorithm implementation"""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def aggregate(self, client_states: List[Dict]) -> None:
        """Aggregate client models"""
        global_state = self.model.state_dict()
        
        for key in global_state.keys():
            # Stack client weights
            client_weights = torch.stack([
                state['model_state'][key].to(self.device) 
                for state in client_states
            ])
            
            global_state[key] = torch.mean(client_weights, dim=0)
        self.model.load_state_dict(global_state)