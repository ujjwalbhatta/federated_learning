import time
import psutil
import numpy as np
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

class DistributedMetricsCollector:
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.round_metrics: List[RoundMetrics] = []
        self.accuracy_metrics = []
        self.system_metrics = {
            'communication_overhead': [],
            'latency': [],
            'throughput': [],
            'cpu_usage': [],
            'memory_usage': [],
            'bandwidth_utilization': [],
            'round_duration': []
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

    def collect_round_metrics(self, round_num: int, accuracy: float, round_start_time: float) -> None:
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

        # Store metrics
        round_metrics = RoundMetrics(
            round_number=round_num,
            accuracy=accuracy,
            training_time=round_duration,
            communication_overhead=bytes_transferred,
            latency=round_duration * 1000,  # Convert to ms
            resource_usage={'cpu': cpu_percent, 'memory': memory_usage},
            throughput=throughput,
            bandwidth_utilization=bandwidth_util
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

    def get_summary_metrics(self) -> Dict:
        return {
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
            },
            'total_time': time.time() - self.start_time
        }

    def save_metrics(self, algorithm_name: str, save_dir: str = 'results/metrics/') -> None:
        os.makedirs(save_dir, exist_ok=True)
        metrics_data = {
            'algorithm': algorithm_name,
            'round_metrics': [vars(m) for m in self.round_metrics],
            'summary': self.get_summary_metrics(),
            'system_metrics': self.system_metrics
        }
        with open(f'{save_dir}{algorithm_name}_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=4)