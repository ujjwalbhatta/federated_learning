import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os

def plot_detailed_performance(all_results: Dict, colors: Dict):
    """Plot detailed performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Performance Metrics', fontsize=16, y=1.02)
    
    algorithms = list(all_results.keys())
    
    # Round Duration Analysis
    for algo, results in all_results.items():
        if 'rounds' in results:
            rounds = range(1, len(results['rounds']) + 1)
            durations = []
            for r in results['rounds']:
                if 'system_metrics' in r and 'system_performance' in r['system_metrics'] and 'round_duration' in r['system_metrics']['system_performance']:
                    durations.append(r['system_metrics']['system_performance']['round_duration']['avg'])
            
            # Ensure rounds and durations have the same length
            rounds = rounds[:len(durations)]
            
            if durations:  # Only plot if durations data exists
                ax1.plot(rounds, durations, 'o-', color=colors[algo], 
                        label=algo, markersize=6, linewidth=2)
    
    ax1.set_title('Round Duration')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Duration (s)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Total Time Comparison - Fixed path
    total_times = [results['system_metrics'].get('total_time', results.get('training_time', 0)) 
                  for results in all_results.values()]
    bars = ax2.bar(algorithms, total_times, color=[colors[algo] for algo in algorithms])
    ax2.set_title('Total Execution Time')
    ax2.set_ylabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Throughput Analysis
    throughput_data = []
    throughput_err = []
    for algo in algorithms:
        tp = all_results[algo]['system_metrics']['system_performance']['throughput']
        throughput_data.append(tp['avg'])
        throughput_err.append([[tp['avg'] - tp['min']], [tp['max'] - tp['avg']]])
    
    bars = ax3.bar(algorithms, throughput_data, color=[colors[algo] for algo in algorithms])
    ax3.set_title('Average Throughput')
    ax3.set_ylabel('Throughput (operations/s)')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Communication Overhead
    comm_data = []
    comm_err = []
    for algo in algorithms:
        comm = all_results[algo]['system_metrics']['system_performance']['communication_overhead']
        comm_data.append(comm['avg'])
        comm_err.append([[comm['avg'] - comm['min']], [comm['max'] - comm['avg']]])
    
    bars = ax4.bar(algorithms, comm_data, color=[colors[algo] for algo in algorithms])
    ax4.set_title('Communication Overhead')
    ax4.set_ylabel('Data Transferred (MB)')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/plots/detailed_performance.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_scalability_metrics(all_results: Dict, colors: Dict):
    """Plot comprehensive scalability metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Scalability and Network Metrics', fontsize=16, y=1.02)
    
    # Completion Time vs Clients
    for algo, results in all_results.items():
        scalability_data = results['distributed_metrics']['scalability']
        if scalability_data:
            num_clients = [data['num_clients'] for data in scalability_data]
            completion_times = [data['completion_time'] for data in scalability_data]
            ax1.plot(num_clients, completion_times, 'o-', color=colors[algo], 
                    label=algo, markersize=6, linewidth=2)
    
    ax1.set_title('Completion Time vs Number of Clients')
    ax1.set_xlabel('Number of Clients')
    ax1.set_ylabel('Completion Time (s)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Throughput vs Clients
    for algo, results in all_results.items():
        scalability_data = results['distributed_metrics']['scalability']
        if scalability_data:
            num_clients = [data['num_clients'] for data in scalability_data]
            throughputs = [data['throughput'] for data in scalability_data]
            ax2.plot(num_clients, throughputs, 'o-', color=colors[algo], 
                    label=algo, markersize=6, linewidth=2)
    
    ax2.set_title('Throughput vs Number of Clients')
    ax2.set_xlabel('Number of Clients')
    ax2.set_ylabel('Throughput')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Network Partition Analysis
    algorithms = list(all_results.keys())
    violations = [all_results[algo]['distributed_metrics']['network_partition'].get('consistency_violations', 0) 
                 for algo in algorithms]
    
    bars = ax3.bar(algorithms, violations, color=[colors[algo] for algo in algorithms])
    ax3.set_title('Consistency Violations During Network Partition')
    ax3.set_ylabel('Number of Violations')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Recovery Analysis
    recovery_times = []
    for algo in algorithms:
        partition_data = all_results[algo]['distributed_metrics']['network_partition']
        recovery_time = partition_data.get('recovery_time', 0)
        recovery_times.append(recovery_time)
    
    bars = ax4.bar(algorithms, recovery_times, color=[colors[algo] for algo in algorithms])
    ax4.set_title('Network Partition Recovery Time')
    ax4.set_ylabel('Recovery Time (s)')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('results/plots/scalability_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

def generate_visualizations(all_results: Dict):
    """Generate comprehensive visualizations from unified metrics"""
    print("\nGenerating visualizations...")
    os.makedirs('results/plots', exist_ok=True)
    
    colors = {
        'fedavg': '#2ecc71',    # green
        'fedprox': '#e74c3c',   # red
        'cwt': '#3498db',       # blue
        'dp_fedavg': '#f1c40f'  # yellow
    }
    
    plt.style.use('classic')
    
    # Generate all plots
    plot_accuracy_comparison(all_results, colors)
    plot_performance_metrics(all_results, colors)
    plot_resource_utilization(all_results, colors)
    plot_network_metrics(all_results, colors)
    plot_detailed_performance(all_results, colors)
    plot_scalability_metrics(all_results, colors)

def plot_accuracy_comparison(all_results: Dict, colors: Dict):
    """Plot accuracy metrics comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Accuracy Metrics Comparison', fontsize=16, y=1.02)
    
    # Global Accuracy Progress
    for algo, results in all_results.items():
        if 'rounds' in results:
            rounds = [r['round'] for r in results['rounds']]
            accuracies = [r['global_accuracy'] for r in results['rounds']]
            ax1.plot(rounds, accuracies, 'o-', color=colors[algo], 
                    label=algo, markersize=6, linewidth=2)
    
    ax1.set_title('Global Accuracy Progress')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Client Accuracy Progress
    for algo, results in all_results.items():
        if 'rounds' in results:
            rounds = range(1, len(results['rounds']) + 1)
            accuracies = [r.get('avg_client_accuracy', 0) for r in results['rounds']]
            ax2.plot(rounds, accuracies, 'o-', color=colors[algo], 
                    label=algo, markersize=6, linewidth=2)
                    
    ax2.set_title('Average Client Accuracy')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Final Accuracy Comparison
    algorithms = list(all_results.keys())
    final_accuracies = [results['final_accuracy'] for results in all_results.values()]
    
    bars = ax3.bar(algorithms, final_accuracies, color=[colors[algo] for algo in algorithms])
    ax3.set_title('Final Global Accuracy')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Consistency Score Progress
    for algo, results in all_results.items():
        if 'rounds' in results:
            rounds = range(1, len(results['rounds']) + 1)
            consistency = [r['consistency_score'] for r in results['rounds']]
            ax4.plot(rounds, consistency, 'o-', color=colors[algo], 
                    label=algo, markersize=6, linewidth=2)
    
    ax4.set_title('Model Consistency Score')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Consistency Score')
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/plots/accuracy_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_performance_metrics(all_results: Dict, colors: Dict):
    """Plot system performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('System Performance Metrics', fontsize=16, y=1.02)
    
    # Training Time per Round
    for algo, results in all_results.items():
        if 'rounds' in results:
            rounds = range(1, len(results['rounds']) + 1)
            times = [r['system_metrics']['latency']/1000 for r in results['rounds']]  # Convert ms to s
            ax1.plot(rounds, times, 'o-', color=colors[algo], 
                    label=algo, markersize=6, linewidth=2)
    
    ax1.set_title('Training Time per Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Time (seconds)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Total Training Time
    algorithms = list(all_results.keys())
    training_times = [results['training_time'] for results in all_results.values()]
    
    bars = ax2.bar(algorithms, training_times, color=[colors[algo] for algo in algorithms])
    ax2.set_title('Total Training Time')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # CPU Usage
    algorithms = list(all_results.keys())
    cpu_data = []
    cpu_err_minus = []
    cpu_err_plus = []
    
    for algo in algorithms:
        data = all_results[algo]['system_metrics']['system_performance']['cpu_usage']
        cpu_data.append(data['avg'])
        cpu_err_minus.append(data['avg'] - data['min'])
        cpu_err_plus.append(data['max'] - data['avg'])
    
    bars = ax3.bar(algorithms, cpu_data, color=[colors[algo] for algo in algorithms])
    ax3.errorbar(algorithms, cpu_data, 
                yerr=[cpu_err_minus, cpu_err_plus],
                fmt='none', color='black', capsize=5)
    
    ax3.set_title('Average CPU Usage')
    ax3.set_ylabel('CPU Usage (%)')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Memory Usage
    memory_data = []
    memory_err_minus = []
    memory_err_plus = []
    
    for algo in algorithms:
        data = all_results[algo]['system_metrics']['system_performance']['memory_usage']
        memory_data.append(data['avg'])
        memory_err_minus.append(data['avg'] - data['min'])
        memory_err_plus.append(data['max'] - data['avg'])
    
    bars = ax4.bar(algorithms, memory_data, color=[colors[algo] for algo in algorithms])
    ax4.errorbar(algorithms, memory_data, 
                yerr=[memory_err_minus, memory_err_plus],
                fmt='none', color='black', capsize=5)
    
    ax4.set_title('Average Memory Usage')
    ax4.set_ylabel('Memory Usage (MB)')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('results/plots/performance_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_resource_utilization(all_results: Dict, colors: Dict):
    """Plot resource utilization metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Resource Utilization Metrics', fontsize=16, y=1.02)
    
    algorithms = list(all_results.keys())
    
    # Bandwidth Utilization
    bandwidth_data = []
    bandwidth_err_minus = []
    bandwidth_err_plus = []
    
    for algo in algorithms:
        bw = all_results[algo]['system_metrics']['system_performance']['bandwidth_utilization']
        bandwidth_data.append(bw['avg'])
        bandwidth_err_minus.append(bw['avg'] - bw['min'])
        bandwidth_err_plus.append(bw['max'] - bw['avg'])
    
    bars = ax1.bar(algorithms, bandwidth_data, color=[colors[algo] for algo in algorithms])
    ax1.errorbar(algorithms, bandwidth_data, 
                yerr=[bandwidth_err_minus, bandwidth_err_plus],
                fmt='none', color='black', capsize=5)
    
    ax1.set_title('Bandwidth Utilization')
    ax1.set_ylabel('Bandwidth (MB/s)')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Communication Overhead
    comm_data = []
    comm_err_minus = []
    comm_err_plus = []
    
    for algo in algorithms:
        comm = all_results[algo]['system_metrics']['system_performance']['communication_overhead']
        comm_data.append(comm['avg'])
        comm_err_minus.append(comm['avg'] - comm['min'])
        comm_err_plus.append(comm['max'] - comm['avg'])
    
    bars = ax2.bar(algorithms, comm_data, color=[colors[algo] for algo in algorithms])
    ax2.errorbar(algorithms, comm_data, 
                yerr=[comm_err_minus, comm_err_plus],
                fmt='none', color='black', capsize=5)
    
    ax2.set_title('Communication Overhead')
    ax2.set_ylabel('Data Transferred (MB)')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/plots/resource_utilization.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_network_metrics(all_results: Dict, colors: Dict):
    """Plot network-related metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Network Performance Metrics', fontsize=16, y=1.02)
    
    algorithms = list(all_results.keys())
    
    # Scalability Analysis
    for algo, results in all_results.items():
        scalability_data = results['distributed_metrics']['scalability']
        if scalability_data:  # Check if data exists
            num_clients = [data['num_clients'] for data in scalability_data]
            completion_times = [data['completion_time'] for data in scalability_data]
            ax1.plot(num_clients, completion_times, 'o-', color=colors[algo], 
                    label=algo, markersize=6, linewidth=2)
    
    ax1.set_title('Scalability Analysis')
    ax1.set_xlabel('Number of Clients')
    ax1.set_ylabel('Completion Time (s)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Consistency Analysis
    for algo, results in all_results.items():
        consistency_data = results['distributed_metrics']['consistency']
        if consistency_data:  # Check if data exists
            rounds = range(1, len(consistency_data) + 1)
            scores = [data['consistency_score'] for data in consistency_data]
            ax2.plot(rounds, scores, 'o-', color=colors[algo], 
                    label=algo, markersize=6, linewidth=2)
    
    ax2.set_title('Consistency Analysis')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Consistency Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Network Partition Recovery
    recovery_times = []
    for algo in algorithms:
        partition_data = all_results[algo]['distributed_metrics'].get('network_partition', {})
        recovery_time = partition_data.get('recovery_time', 0)
        recovery_times.append(recovery_time)
    
    bars = ax3.bar(algorithms, recovery_times, color=[colors[algo] for algo in algorithms])
    ax3.set_title('Network Partition Recovery Time')
    ax3.set_ylabel('Recovery Time (s)')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Message Delivery Rate
    delivery_rates = []
    for algo in algorithms:
        resilience_data = all_results[algo]['distributed_metrics']['network_resilience']
        delivery_rate = resilience_data.get('message_delivery_rate', 0)
        delivery_rates.append(delivery_rate)
    
    bars = ax4.bar(algorithms, delivery_rates, color=[colors[algo] for algo in algorithms])
    ax4.set_title('Message Delivery Rate')
    ax4.set_ylabel('Delivery Rate (%)')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('results/plots/network_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()