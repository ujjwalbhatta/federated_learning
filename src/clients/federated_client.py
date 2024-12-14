import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

class FederatedClient:
    def __init__(self, model: nn.Module, device: torch.device, client_id: int,
                 learning_rate: float = 0.01, local_epochs: int = 5, algorithm: str = 'fedavg',
                 mu: float = 0.01):  # mu for FedProx
        self.model = model
        self.device = device
        self.client_id = client_id
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.algorithm = algorithm
        self.mu = mu

    def train(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Train the local model with algorithm-specific modifications"""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Store initial model state for FedProx
        if self.algorithm == 'fedprox':
            initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        total_loss = 0
        num_samples = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                num_samples += len(labels)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Add proximal term for FedProx
                if self.algorithm == 'fedprox':
                    proximal_term = 0
                    for param, init_param in zip(self.model.parameters(), initial_state.values()):
                        proximal_term += (self.mu / 2) * torch.norm(param - init_param.to(self.device)) ** 2
                    loss += proximal_term

                if self.algorithm == 'dp_fedavg':
                    l2_norm = sum(torch.norm(p) ** 2 for p in self.model.parameters())
                    loss += 0.01 * l2_norm  # Small L2 regularization

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            total_loss += epoch_loss / len(dataloader)

        return {
            'model_state': self.model.state_dict(),
            'average_loss': total_loss / self.local_epochs,
            'num_samples': num_samples  # Added for CWT
        }

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate the local model"""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return {
            'accuracy': correct / total,
            'loss': total_loss / len(dataloader)
        }