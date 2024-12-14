import torch
import logging
from datetime import datetime
import os
import json
import time
from src.models.mnist_model import MNISTModel
from src.clients.federated_client import FederatedClient
from src.utils.data_loader import load_mnist_data
from src.utils.distributed_metrics import DistributedMetricsCollector
from src.utils.visualization_utils import generate_visualizations
from src.algorithms.fedavg import FedAvg
from src.algorithms.fedprox import FedProx
from src.algorithms.cwt import CyclicWeightTransfer
from src.algorithms.dp_fedavg import DPFedAvg

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/federated_learning_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler()
        ]
    )

def run_experiment(algorithm_name: str, config: dict):
    metrics_collector = DistributedMetricsCollector(config['num_clients'])
    device = torch.device(config['device'])
    
    logging.info(f"Initializing experiment with {algorithm_name}")
    logging.info(f"Using device: {device}")
    
    # Initialize algorithm
    logging.info("Initializing global model")
    global_model = MNISTModel().to(device)
    if algorithm_name == 'fedavg':
        algorithm = FedAvg(global_model, device)
    elif algorithm_name == 'fedprox':
        algorithm = FedProx(global_model, device, mu=config.get('mu', 0.01))
    elif algorithm_name == 'cwt':
        algorithm = CyclicWeightTransfer(global_model, device)
    elif algorithm_name == 'dp_fedavg':
        algorithm = DPFedAvg(global_model, device, epsilon=config.get('epsilon', 5.0))
    
    # Load data
    logging.info("Loading and preparing data")
    client_loaders, test_loader = load_mnist_data(config['num_clients'], config['batch_size'])
    logging.info(f"Data loaded successfully. Number of clients: {config['num_clients']}")
    
    # Initialize clients
    logging.info("Initializing clients")
    clients = [
        FederatedClient(
            MNISTModel().to(device),
            device,
            client_id=i,
            learning_rate=config['learning_rate'],
            local_epochs=config['local_epochs'],
            algorithm=algorithm_name
        ) for i in range(config['num_clients'])
    ]
    logging.info(f"Initialized {len(clients)} clients successfully")
    
    results = {
        'algorithm': algorithm_name,
        'rounds': [],
        'final_accuracy': 0,
        'training_time': 0,
        'system_metrics': {}
    }
    
    start_time = datetime.now()
    
    for round_num in range(config['num_rounds']):
        round_start_time = time.time()
        logging.info(f"\nStarting round {round_num + 1}/{config['num_rounds']}")
        client_states = []
        round_accuracies = []
        
        for client_id, (client, loader) in enumerate(zip(clients, client_loaders)):
            logging.info(f"Training client {client_id + 1}/{config['num_clients']} in round {round_num + 1}")
            
            try:
                # Train client
                client.model.load_state_dict(algorithm.model.state_dict())
                training_result = client.train(loader)
                client_states.append(training_result)
                
                # Evaluate client model
                metrics = client.evaluate(loader)
                round_accuracies.append(metrics['accuracy'])
                logging.info(f"Round {round_num + 1}, Client {client_id + 1} training completed. "
                           f"Accuracy: {metrics['accuracy']:.4f}")
            
            except Exception as e:
                logging.error(f"Error training client {client_id + 1}: {str(e)}")
                logging.error("Stack trace:", exc_info=True)
                continue
        
        try:
            logging.info(f"Aggregating models for round {round_num + 1}")
            algorithm.aggregate(client_states)
            
            # Evaluate global model
            global_accuracy = evaluate_model(algorithm.model, test_loader, device)
            logging.info(f"Round {round_num + 1} completed. Global accuracy: {global_accuracy:.4f}")
            
            # Collect metrics
            metrics_collector.collect_round_metrics(
                round_num + 1,
                global_accuracy,
                round_start_time
            )
            
            results['rounds'].append({
                'round': round_num + 1,
                'global_accuracy': global_accuracy,
                'avg_client_accuracy': sum(round_accuracies) / len(round_accuracies),
                'round_time': time.time() - round_start_time
            })
            
        except Exception as e:
            logging.error(f"Error in round {round_num + 1}: {str(e)}")
            logging.error("Stack trace:", exc_info=True)
            continue
    
    metrics_collector.save_metrics(algorithm_name)
    
    results['final_accuracy'] = global_accuracy
    results['training_time'] = (datetime.now() - start_time).total_seconds()
    results['system_metrics'] = metrics_collector.get_summary_metrics()
    
    save_results(results, algorithm_name)
    logging.info(f"Experiment with {algorithm_name} completed successfully")
    
    return results

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def save_results(results, algorithm_name):
    os.makedirs('results', exist_ok=True)
    with open(f'results/{algorithm_name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

def main():
    try:
        setup_logging()
        config = {
            'num_clients': 5,
            'num_rounds': 10,
            'batch_size': 32,
            'learning_rate': 0.01,
            'local_epochs': 5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mu': 0.01,
            'epsilon': 1.0
        }
        
        algorithms = ['fedavg', 'fedprox', 'cwt', 'dp_fedavg']
        all_results = {}
        
        logging.info("Starting experiments...")
        print("Starting experiments...")
        
        for algorithm in algorithms:
            logging.info(f"Starting experiment with {algorithm}")
            print(f"Starting experiment with {algorithm}")
            results = run_experiment(algorithm, config)
            all_results[algorithm] = results
            logging.info(f"Completed experiment with {algorithm}")
        
        generate_visualizations(all_results)
        logging.info("All experiments completed successfully")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()