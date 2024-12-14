import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os

def generate_visualizations(all_results: Dict):
    """Generate comprehensive visualizations including both accuracy and system metrics"""
    # Create results directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)
    
    print("\nGenerating visualizations...")
    # Plot accuracy comparison
    plot_accuracy_comparison(all_results)
    print("Generated accuracy comparison plot")
    
    # Plot system metrics
    plot_system_metrics(all_results)
    print("Generated system metrics plots")
    
    # Plot resource utilization
    plot_resource_utilization(all_results)
    print("Generated resource utilization plots")
    print("\nAll plots have been saved to results/plots/")

def plot_accuracy_comparison(all_results: Dict):
    """Plot accuracy comparison between algorithms"""
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm']  # Different colors for different algorithms
    
    for idx, (algo, results) in enumerate(all_results.items()):
        if 'rounds' in results:
            rounds = [r['round'] for r in results['rounds']]
            accuracies = [r['global_accuracy'] for r in results['rounds']]
            plt.plot(rounds, accuracies, f'{colors[idx%4]}o-', 
                    label=algo, linewidth=2, markersize=8)
    
    plt.title('Algorithm Comparison - Global Model Accuracy', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = 'results/plots/accuracy_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_system_metrics(all_results: Dict):
    """Plot system performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['b', 'g', 'r', 'm']
    
    for idx, (algo, results) in enumerate(all_results.items()):
        if 'system_metrics' in results:
            metrics = results['system_metrics']['system_performance']
            rounds = range(1, len(results['rounds']) + 1)
            color = colors[idx % 4]
            
            # Throughput
            ax1.plot(rounds, [metrics['throughput']['avg']] * len(rounds), 
                    f'{color}o-', label=algo, linewidth=2)
            
            # Communication Overhead
            ax2.plot(rounds, [metrics['communication_overhead']['avg']] * len(rounds), 
                    f'{color}o-', label=algo, linewidth=2)
            
            # Latency
            ax3.plot(rounds, [metrics['latency']['avg']] * len(rounds), 
                    f'{color}o-', label=algo, linewidth=2)
            
            # Bandwidth
            ax4.plot(rounds, [metrics['bandwidth_utilization']['avg']] * len(rounds), 
                    f'{color}o-', label=algo, linewidth=2)
    
    ax1.set_title('System Throughput', fontsize=12)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Operations/second')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Communication Overhead', fontsize=12)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('MB transferred')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('System Latency', fontsize=12)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Milliseconds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('Bandwidth Utilization', fontsize=12)
    ax4.set_xlabel('Round')
    ax4.set_ylabel('MB/s')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'results/plots/system_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_resource_utilization(all_results: Dict):
    """Plot resource utilization metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['b', 'g', 'r', 'm']
    
    for idx, (algo, results) in enumerate(all_results.items()):
        if 'system_metrics' in results:
            metrics = results['system_metrics']['system_performance']
            rounds = range(1, len(results['rounds']) + 1)
            color = colors[idx % 4]
            
            # CPU Usage
            ax1.plot(rounds, [metrics['cpu_usage']['avg']] * len(rounds), 
                    f'{color}o-', label=algo, linewidth=2)
            
            # Memory Usage
            ax2.plot(rounds, [metrics['memory_usage']['avg']] * len(rounds), 
                    f'{color}o-', label=algo, linewidth=2)
    
    ax1.set_title('CPU Usage', fontsize=12)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('CPU %')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Memory Usage', fontsize=12)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Memory (MB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'results/plots/resource_utilization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()