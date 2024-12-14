from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List, Dict

class FederatedAlgorithm(ABC):
    """Base class for federated learning algorithms"""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @abstractmethod
    def aggregate(self, client_states: List[Dict]) -> None:
        """Aggregate client models"""
        pass
