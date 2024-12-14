import torch
import torch.nn as nn
from typing import List, Dict

class DPFedAvg:
    """
    Simplified Differentially Private Federated Averaging
    with more conservative privacy mechanisms
    """
    def __init__(self, model: nn.Module, device: torch.device, epsilon: float = 1.0):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        # More conservative parameters
        self.noise_scale = 0.01  # Much smaller noise scale
        self.clip_norm = 5.0     # Larger clipping threshold

    def clip_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Clip weights with a larger threshold"""
        norm = torch.norm(weights)
        if norm > self.clip_norm:
            return weights * (self.clip_norm / norm)
        return weights

    def add_noise(self, tensor: torch.Tensor, num_clients: int) -> torch.Tensor:
        """Add scaled Gaussian noise"""
        # Scale noise based on number of clients and epsilon
        scaled_noise = self.noise_scale / (num_clients * self.epsilon)
        noise = torch.randn_like(tensor) * scaled_noise
        return tensor + noise

    def aggregate(self, client_states: List[Dict]) -> None:
        """Aggregate with minimal privacy mechanism"""
        global_state = self.model.state_dict()
        num_clients = len(client_states)

        for key in global_state.keys():
            # Get all client weights for this layer
            client_weights = torch.stack([
                state['model_state'][key].to(self.device) 
                for state in client_states
            ])

            # Calculate average
            avg_weights = torch.mean(client_weights, dim=0)

            # Clip the averaged weights
            clipped_weights = self.clip_weights(avg_weights)

            # Add minimal noise
            noisy_weights = self.add_noise(clipped_weights, num_clients)

            # Update global model
            global_state[key] = noisy_weights

        self.model.load_state_dict(global_state)