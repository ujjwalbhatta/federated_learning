from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import List, Tuple

def load_mnist_data(num_clients: int, batch_size: int) -> Tuple[List[DataLoader], DataLoader]:
    """Load and split MNIST dataset for federated learning."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training data among clients
    data_per_client = len(train_dataset) // num_clients
    client_datasets = random_split(train_dataset, [data_per_client] * num_clients)
    
    client_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in client_datasets
    ]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader
