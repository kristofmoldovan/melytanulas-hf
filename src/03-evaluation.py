# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
from utils import setup_logger, create_deterministic_splitted_datasets
import config
from lib.dataloader import FlagDataset
from lib.baseline_model import BaselineClassifier
from lib.model import FlagClassifier
from lib.seed_everything import seed_everything
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

logger = setup_logger()

def evaluate():
    logger.info("Evaluating model...")

    full_dataset = FlagDataset(csv_file=CSV_FILE, target_length=FLAG_TARGET_LENGTH)
    train_ds, val_ds, test_ds = create_deterministic_splitted_datasets(full_dataset)

    # Loading models
    model = FlagClassifier(num_classes=len(full_dataset.classes))
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=config.DEVICE))

    baseline = BaselineClassifier(num_classes=len(full_dataset.classes))
    baseline.load_state_dict(torch.load(BASELINE_LOAD_PATH, map_location=config.DEVICE))

    logger.info("Models loaded successfully.")
    #paths
    logger.info(f"Model loaded from: {MODEL_LOAD_PATH}")
    logger.info(f"Baseline model loaded from: {BASELINE_LOAD_PATH}")

    # Only using test_ds for evaluation
    logger.info(f"Test dataset size: {len(test_ds)} samples.")

    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)



    all_histories = {}
    trained_models = {}


    #all_histories['Baseline History'] = baseline_history
    trained_models['BaselineModel'] = baseline

    #all_histories['Model History'] = model_history
    trained_models['Model'] = model

    # --- 3. Visualize & Benchmark ---

    # A. Plot Learning Curves
    #plot_comparison(all_histories)

    # B. Run Inference Benchmark (Speed vs Accuracy)
    print("\n--- Final Benchmark Results ---")
    # Note: Using val_loader as test_loader for this example
    stats = benchmark_models(trained_models, test_loader, device=DEVICE)
    print(stats)

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

if __name__ == "__main__":

    CSV_FILE = config.CSV_FILE
    FLAG_TARGET_LENGTH = config.FLAG_TARGET_LENGTH
    MODEL_LOAD_PATH = config.MODEL_LOAD_PATH
    BASELINE_LOAD_PATH = config.BASELINE_LOAD_PATH
    BATCH_SIZE = config.BATCH_SIZE
    DEVICE = config.DEVICE

    seed_everything(config.SEED)

    evaluate()
