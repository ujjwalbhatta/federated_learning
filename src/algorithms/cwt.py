import torch
import torch.nn as nn
from typing import List, Dict

class CyclicWeightTransfer:
    """
    Cyclic Weight Transfer (CWT) implementation
    Transfers model weights between clients in a ring topology
    """
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.current_client_idx = 0  # Track current client in the cycle

    def aggregate(self, client_states: List[Dict]) -> None:
        """
        In CWT, we don't actually aggregate - we just take the weights 
        from the current client and pass them to the next
        """
        # Get the next client's weights in the cycle
        next_client_idx = (self.current_client_idx + 1) % len(client_states)
        
        # Load weights from the current client
        current_state = client_states[self.current_client_idx]['model_state']
        self.model.load_state_dict(current_state)
        
        # Update current client index for next round
        self.current_client_idx = next_client_idx

    def get_next_client_id(self, num_clients: int) -> int:
        """Get the ID of the next client in the cycle"""
        return (self.current_client_idx + 1) % num_clients