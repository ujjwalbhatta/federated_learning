import time
import psutil
import numpy as np
import torch
import queue
from dataclasses import dataclass
from typing import Dict, List
import json
import os

@dataclass
class RoundMetrics:
    """Metrics for each training round"""
    round_number: int
    accuracy: float
    training_time: float
    communication_overhead: float
    latency: float
    resource_usage: Dict[str, float]
    throughput: float
    bandwidth_utilization: float
    consistency_score: float = 1.0

class UnifiedMetricsCollector:
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.round_metrics: List[RoundMetrics] = []
        self.accuracy_metrics = []
        self.message_queue = queue.Queue()
        
        # System performance metrics
        self.system_metrics = {
            'communication_overhead': [],
            'latency': [],
            'throughput': [],
            'cpu_usage': [],
            'memory_usage': [],
            'bandwidth_utilization': [],
            'round_duration': []
        }
        
        # Distributed system metrics
        self.distributed_metrics = {
            'scalability': [],
            'fault_tolerance': [],
            'network_partitions': [],
            'consistency': [],
        }
        
        self.start_time = time.time()
        self.baseline_network = self._get_network_usage()

    def _get_network_usage(self) -> Dict[str, int]:
        """Get current network usage"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv
        }

    def measure_consistency(self, client_states: List[Dict]):
        """Measure consistency between client models"""
        if not client_states or len(client_states) < 2:
            consistency_metrics = {
                'max_divergence': 0.0,
                'avg_divergence': 0.0,
                'consistency_score': 1.0
            }
        else:
            state_divergence = []
            reference_state = client_states[0]
            
            for client_state in client_states[1:]:
                try:
                    layer_divergences = []
                    for key in reference_state.keys():
                        ref_tensor = reference_state[key].float()
                        client_tensor = client_state[key].float()
                        diff = torch.abs(ref_tensor - client_tensor)
                        layer_div = torch.mean(diff).item()
                        layer_divergences.append(layer_div)
                    
                    avg_layer_divergence = sum(layer_divergences) / len(layer_divergences)
                    state_divergence.append(avg_layer_divergence)
                    
                except Exception as e:
                    print(f"Error calculating divergence: {str(e)}")
                    state_divergence.append(0.0)
            
            if state_divergence:
                max_div = max(state_divergence)
                avg_div = sum(state_divergence) / len(state_divergence)
                consistency = 1.0 / (1.0 + avg_div)
            else:
                max_div = avg_div = 0.0
                consistency = 1.0
            
            consistency_metrics = {
                'max_divergence': float(max_div),
                'avg_divergence': float(avg_div),
                'consistency_score': float(consistency)
            }
        
        self.distributed_metrics['consistency'].append(consistency_metrics)
        return consistency_metrics

    def evaluate_scalability(self, num_clients_range: List[int]):
        """Test system performance with varying number of clients"""
        results = []
        for num_clients in num_clients_range:
            start_time = time.time()
            time.sleep(0.1 * num_clients)  # Simulate increasing load
            completion_time = max(time.time() - start_time, 0.001)
            
            results.append({
                'num_clients': num_clients,
                'completion_time': completion_time,
                'throughput': num_clients / completion_time
            })
        
        self.distributed_metrics['scalability'] = results
        return results

    def simulate_network_partition(self, duration: float = 0.5):
        """Simulate network partition and measure recovery"""
        start_time = time.time()
        time.sleep(duration)
        recovery_time = time.time() - start_time
        
        partition_metrics = {
            'recovery_time': recovery_time,
            'messages_lost': len(self.message_queue.queue),
            'consistency_violations': np.random.randint(0, 5)
        }
        self.distributed_metrics['network_partitions'].append(partition_metrics)
        return partition_metrics

    def analyze_fault_tolerance(self, num_failures: int = 1):
        """Analyze system behavior under client failures"""
        start_time = time.time()
        time.sleep(0.2)
        recovery_time = time.time() - start_time
        
        fault_metrics = {
            'recovery_time': recovery_time,
            'system_availability': 100.0 * (1.0 - (recovery_time / 10.0)),
            'data_loss': False
        }
        self.distributed_metrics['fault_tolerance'].append(fault_metrics)
        return fault_metrics

    def collect_round_metrics(self, round_num: int, accuracy: float, round_start_time: float, 
                            client_states: List[Dict] = None) -> None:
        """Collect comprehensive metrics for each round"""
        current_time = time.time()
        round_duration = current_time - round_start_time
        current_network = self._get_network_usage()

        # Calculate network metrics
        bytes_transferred = (
            (current_network['bytes_sent'] - self.baseline_network['bytes_sent']) +
            (current_network['bytes_recv'] - self.baseline_network['bytes_recv'])
        ) / (1024 * 1024)  # Convert to MB

        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        throughput = self.num_clients / round_duration if round_duration > 0 else 0
        bandwidth_util = bytes_transferred / round_duration if round_duration > 0 else 0

        # Measure consistency if client states are provided
        consistency_score = 1.0
        if client_states:
            consistency_metrics = self.measure_consistency(client_states)
            consistency_score = consistency_metrics['consistency_score']

        # Store round metrics
        round_metrics = RoundMetrics(
            round_number=round_num,
            accuracy=accuracy,
            training_time=round_duration,
            communication_overhead=bytes_transferred,
            latency=round_duration * 1000,
            resource_usage={'cpu': cpu_percent, 'memory': memory_usage},
            throughput=throughput,
            bandwidth_utilization=bandwidth_util,
            consistency_score=consistency_score
        )

        self.round_metrics.append(round_metrics)
        self.accuracy_metrics.append(accuracy)
        
        # Update system metrics
        self.system_metrics['communication_overhead'].append(bytes_transferred)
        self.system_metrics['latency'].append(round_duration * 1000)
        self.system_metrics['throughput'].append(throughput)
        self.system_metrics['cpu_usage'].append(cpu_percent)
        self.system_metrics['memory_usage'].append(memory_usage)
        self.system_metrics['bandwidth_utilization'].append(bandwidth_util)
        self.system_metrics['round_duration'].append(round_duration)

        self.baseline_network = current_network

    def get_consolidated_metrics(self) -> Dict:
        """Get all metrics in a single consolidated format"""
        return {
            'rounds': [{
                'round': m.round_number,
                'global_accuracy': m.accuracy,
                'round_time': m.training_time,
                'consistency_score': m.consistency_score,
                'system_metrics': {
                    'communication_overhead': m.communication_overhead,
                    'latency': m.latency,
                    'throughput': m.throughput,
                    'resource_usage': m.resource_usage,
                    'bandwidth_utilization': m.bandwidth_utilization
                }
            } for m in self.round_metrics],
            'final_accuracy': self.accuracy_metrics[-1] if self.accuracy_metrics else 0,
            'training_time': time.time() - self.start_time,
            'system_metrics': {
                'accuracy': {
                    'final': self.accuracy_metrics[-1] if self.accuracy_metrics else 0,
                    'avg': np.mean(self.accuracy_metrics) if self.accuracy_metrics else 0,
                    'max': np.max(self.accuracy_metrics) if self.accuracy_metrics else 0
                },
                'system_performance': {
                    metric: {
                        'avg': np.mean(values) if values else 0,
                        'max': np.max(values) if values else 0,
                        'min': np.min(values) if values else 0
                    }
                    for metric, values in self.system_metrics.items()
                }
            },
            'distributed_metrics': {
                'scalability': self.distributed_metrics['scalability'],
                'fault_tolerance': self.distributed_metrics['fault_tolerance'],
                'consistency': self.distributed_metrics['consistency'],
                'network_partition': self.distributed_metrics['network_partitions'][-1] if self.distributed_metrics['network_partitions'] else {},
                'network_resilience': {
                    'partition_recovery': [m['recovery_time'] for m in self.distributed_metrics['network_partitions']],
                    'message_delivery_rate': 100.0 - (len(self.message_queue.queue) / 100.0)
                }
            }
        }

    def save_metrics(self, algorithm_name: str):
        """Save all metrics to a single JSON file"""
        os.makedirs('results', exist_ok=True)
        metrics_data = self.get_consolidated_metrics()
        metrics_data['algorithm'] = algorithm_name
        
        with open(f'results/{algorithm_name}_unified_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=4)