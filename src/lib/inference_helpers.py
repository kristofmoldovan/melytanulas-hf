# Setup plot methods
import matplotlib.pyplot as plt
import torch
import time
import numpy as np

def plot_comparison(histories):
    """
    Plots Train/Val Loss and Accuracy for multiple models.
    
    Args:
        histories (dict): { 'ModelName': history_dict, ... }
    """
    plt.figure(figsize=(12, 5))

    # --- Plot Loss ---
    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        # Plot Validation Loss (Solid line)
        plt.plot(hist['val_loss'], label=f"{name} Val", linestyle='-')
        # Plot Training Loss (Dashed line, lighter)
        plt.plot(hist['train_loss'], label=f"{name} Train", linestyle='--', alpha=0.5)
    
    plt.title("Loss Curves (Lower is better)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Plot Accuracy ---
    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist['val_acc'], label=f"{name} Val", linestyle='-')
        # Optional: Plot Train Acc if you want to check overfitting
        # plt.plot(hist['train_acc'], label=f"{name} Train", linestyle='--', alpha=0.5)

    plt.title("Accuracy Curves (Higher is better)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def benchmark_models(models_dict, test_loader, device='cpu'):
    """
    Runs inference on a test set to measure Accuracy, Latency, and Model Size.
    """
    results = []
    
    print(f"{'Model Name':<20} | {'Params':<10} | {'Acc':<8} | {'Latency (ms)':<15}")
    print("-" * 65)

    for name, model in models_dict.items():
        model = model.to(device)
        model.eval()
        
        # 1. Count Parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # 2. Run Inference & Time it
        correct = 0
        total = 0
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        end_time = time.time()
        
        # 3. Calculate Metrics
        accuracy = correct / total
        total_time_ms = (end_time - start_time) * 1000
        avg_latency = total_time_ms / total # Time per single sample
        
        print(f"{name:<20} | {num_params:<10,} | {accuracy:.2%} | {avg_latency:.4f} ms")
        
        results.append({
            'name': name,
            'params': num_params,
            'accuracy': accuracy,
            'latency': avg_latency
        })
        
    return results