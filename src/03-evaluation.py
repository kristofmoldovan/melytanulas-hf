# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
import seaborn as sns
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
from utils import setup_logger, create_deterministic_splitted_datasets

import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix

logger = setup_logger()

def evaluate():
    logger.info("Evaluating model...")

    full_dataset = FlagDataset(csv_file=CSV_FILE, target_length=FLAG_TARGET_LENGTH, return_idx=True)
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

    baseline_history = pd.read_csv(config.BASELINE_TRAINING_HISTORY_PATH, index_col=0)
    model_history = pd.read_csv(config.MODEL_TRAINING_HISTORY_PATH, index_col=0)


    all_histories = {}
    trained_models = {}


    all_histories['Baseline History'] = baseline_history
    trained_models['BaselineModel'] = baseline

    all_histories['Model History'] = model_history
    trained_models['Model'] = model

    # --- 3. Visualize & Benchmark ---

    # A. Plot Learning Curves
    plot_comparison(all_histories)

    # B. Run Inference Benchmark (Speed vs Accuracy)
    
    print("\n--- Test dataset Accuracy & Latency---")
    # Note: Using val_loader as test_loader for this example
    stats = benchmark_models(trained_models, test_loader, device=DEVICE)
    
    #print(stats)

    # Running more detailed evaluation

    logger.info("\n--- Detailed Evaluation for Baseline ---")
    evaluate_model("BASELINE", baseline, full_dataset.classes, train_ds, val_ds, test_ds, batch_size=config.BATCH_SIZE, device=DEVICE)
    print("\n\n\n---------\n\n\n")
    logger.info("\n--- Detailed Evaluation for Model ---")
    evaluate_model("BUILT MODEL", model, full_dataset.classes, train_ds, val_ds, test_ds, batch_size=config.BATCH_SIZE, device=DEVICE)

    logger.info("\n--- Directional Accuracy Evaluation for Baseline ---")

    evaluate_directional_accuracy("BASELINE", baseline, train_ds, val_ds, test_ds)


    logger.info("\n--- Directional Accuracy Evaluation for Built Model ---")

    evaluate_directional_accuracy("BUILT MODEL", model, train_ds, val_ds, test_ds)


def plot_comparison(histories):
    """
    Plots Train/Val Loss and Accuracy for multiple models.

    Args:
        histories (dict): { 'ModelName': history_dict, ... }
    """
    plt.figure(figsize=(12, 5))

    # --- Plot Loss ---
    #plt.subplot(1, 2, 1)
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
    #plt.subplot(1, 2, 2)
    #for name, hist in histories.items():
        #plt.plot(hist['val_acc'], label=f"{name} Val", linestyle='-')
        # Optional: Plot Train Acc if you want to check overfitting
        # plt.plot(hist['train_acc'], label=f"{name} Train", linestyle='--', alpha=0.5)

   #plt.title("Accuracy Curves (Higher is better)")
    #plt.xlabel("Epochs")
    #plt.ylabel(":")
    #plt.legend()
    #plt.grid(True, alpha=0.3)

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


def evaluate_model(tag, model, classes, train_ds, val_ds, test_ds, batch_size=64, device=None):
    """
    Evaluates a PyTorch model on Train, Val, and Test datasets.
    Prints full text reports (Global, Confusion Matrix, Per-Class) and plots the results.
    """
    
    # --- 1. SETUP & VALIDATION ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
    
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    datasets = {"Train": train_ds, "Val": val_ds, "Test": test_ds}
    num_classes = len(classes)
    
    # Validate Datasets
    for name, ds in datasets.items():
        if not isinstance(ds, Dataset):
             if not (hasattr(ds, '__len__') and hasattr(ds, '__getitem__')):
                raise TypeError(f"Error: {name}_ds is not a valid PyTorch Dataset.")
        if len(ds) == 0:
            raise ValueError(f"Error: {name}_ds is empty.")
        try:
            # Smoke test
            _ = ds[0] 
        except Exception as e:
            raise RuntimeError(f"Failed to fetch item from {name}_ds. Error: {e}")

    # --- 2. INITIALIZE METRICS ---
    # added Top-3 Accuracy here
    metrics = torchmetrics.MetricCollection({
        'Global Acc': torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        'Top-2 Acc':  torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=2),
        'Top-3 Acc':  torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3),
        'Precision':  torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=None),
        'Recall':     torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=None),
        'ConfMat':    torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
    }).to(device)

    results_list = []

    # --- 3. EVALUATION LOOP ---
    with torch.no_grad():
        for phase, ds in datasets.items():
            print(f"\n{'='*20} {phase.upper()} DATASET {'='*20}")
            
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            
            # Inference Loop
            for inputs, targets, _ in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                metrics.update(logits, targets)
            
            # Compute final results for this dataset
            res = metrics.compute()
            
            # --- A. PRINT GLOBAL METRICS ---
            print(f"\n[Global Metrics]")
            print(f"Accuracy:      {res['Global Acc'].item():.4f}")
            print(f"Top-2 Acc:     {res['Top-2 Acc'].item():.4f}")
            print(f"Top-3 Acc:     {res['Top-3 Acc'].item():.4f}")

            # --- B. PRINT CONFUSION MATRIX ---
            cm = res['ConfMat'].cpu()
            #print(f"\n[Confusion Matrix] (Rows=True, Cols=Pred)")
            # Create a Pandas DataFrame for a prettier print
            #cm_df = pd.DataFrame(cm.numpy(), index=classes, columns=classes)
            #print(cm_df)

            # --- C. CALCULATE & PRINT PER-CLASS METRICS ---
            total_samples = cm.sum()
            per_class_acc = []
            
            prec_vals = res['Precision'].cpu().tolist()
            rec_vals = res['Recall'].cpu().tolist()

            print(f"\n[Per-Class Metrics]")
            print(f"{'Class Name':<15} | {'Precision':<10} | {'Recall':<10} | {'OvR Acc':<10}")
            print("-" * 55)

            for i, class_name in enumerate(classes):
                # Custom Calculation: One-vs-Rest Accuracy
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = total_samples - (tp + fp + fn)
                acc_ovr = (tp + tn) / total_samples
                per_class_acc.append(acc_ovr.item())
                
                # Print Row
                print(f"{class_name:<15} | {prec_vals[i]:.4f}     | {rec_vals[i]:.4f}     | {acc_ovr:.4f}")

                # Store for plotting
                results_list.append({"Dataset": phase, "Class": class_name, "Metric": "Precision", "Score": prec_vals[i]})
                results_list.append({"Dataset": phase, "Class": class_name, "Metric": "Recall", "Score": rec_vals[i]})
                results_list.append({"Dataset": phase, "Class": class_name, "Metric": "OvR Accuracy", "Score": acc_ovr.item()})

            # Reset for next dataset
            metrics.reset()

    # --- 4. PLOTTING ---
    df = pd.DataFrame(results_list)
    
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df, x="Class", y="Score", hue="Metric", col="Dataset", 
        kind="bar", palette="viridis", height=5, aspect=1.2, sharey=True
    )
    g.set_axis_labels("", "Score (0-1)")
    g.set(ylim=(0, 1.05))
    g.fig.suptitle(f'{tag} - Evaluation Metrics by Class and Dataset', fontsize=16, y=1.05)
    
    # Rotate labels
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.show()
    
    return df

def evaluate_directional_accuracy(tag, model, train_ds, val_ds, test_ds, batch_size=64, device=None):
    """
    Evaluates Directional Accuracy.
    Group 0 (Bearish): ['Bearish Normal', 'Bearish Pennant', 'Bearish Wedge'] (Indices 0-2)
    Group 1 (Bullish): ['Bullish Normal', 'Bullish Pennant', 'Bullish Wedge'] (Indices 3-5)
    """
    # 1. SETUP
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
            
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    datasets = {"Train": train_ds, "Val": val_ds, "Test": test_ds}
    
    # Metrics (Binary because we are grouping into 2 sets)
    acc_metric = BinaryAccuracy().to(device)
    conf_mat = BinaryConfusionMatrix().to(device)
    
    results_list = []

    print(f"\n{'='*20} DIRECTIONAL EVALUATION {'='*20}")
    print("Mapping: [0,1,2] -> Bearish (Down) | [3,4,5] -> Bullish (Up)")

    with torch.no_grad():
        for phase, ds in datasets.items():
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            
            all_preds_bin = []
            all_targets_bin = []
            
            for inputs, targets, _ in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward Pass
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                
                # --- MAPPING LOGIC ---
                # Indices 0, 1, 2 are < 3 (Bearish Group -> 0)
                # Indices 3, 4, 5 are >= 3 (Bullish Group -> 1)
                preds_bin = (preds >= 3).long()
                targets_bin = (targets >= 3).long()
                
                all_preds_bin.append(preds_bin)
                all_targets_bin.append(targets_bin)
            
            # Combine all batches
            preds_full = torch.cat(all_preds_bin)
            targets_full = torch.cat(all_targets_bin)
            
            # Compute Metrics
            acc = acc_metric(preds_full, targets_full)
            cm = conf_mat(preds_full, targets_full).cpu().numpy()
            
            # Extract CM stats
            # [0,0]=TN (True Bearish), [0,1]=FP (False Bullish)
            # [1,0]=FN (False Bearish), [1,1]=TP (True Bullish)
            tn, fp = cm[0] 
            fn, tp = cm[1] 
            total = cm.sum()
            
            # Key Ratios
            accuracy = (tn + tp) / total
            critical_error_rate = fp / total  # Predicted Bullish, actually Bearish (Loss)
            missed_opp_rate = fn / total      # Predicted Bearish, actually Bullish (FOMO)
            
            # Print Text Report
            print(f"\n--- {phase} DATASET ---")
            print(f"Directional Acc: {accuracy:.2%}")
            print(f"Critical Error:  {critical_error_rate:.2%} (Pred Bullish but was Bearish)")
            
            # Store for Plotting
            results_list.append({"Dataset": phase, "Metric": "Directional Acc", "Score": accuracy})
            results_list.append({"Dataset": phase, "Metric": "Critical Error (Risk)", "Score": critical_error_rate})
            results_list.append({"Dataset": phase, "Metric": "Missed Opp (FOMO)", "Score": missed_opp_rate})

            # Reset metrics for next loop
            acc_metric.reset()
            conf_mat.reset()

    # 2. PLOTTING
    df = pd.DataFrame(results_list)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    ax = sns.barplot(
        data=df, 
        x="Dataset", 
        y="Score", 
        hue="Metric", 
        palette=["#2ecc71", "#e74c3c", "#f1c40f"] # Green, Red, Yellow
    )
    
    # Formatting
    ax.set_title(f"{tag} - Bullish/Bearish Directional Performance", fontsize=16)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Percentage (0-1)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f')

    plt.tight_layout()
    plt.show()
    
    return df

if __name__ == "__main__":

    CSV_FILE = config.CSV_FILE
    FLAG_TARGET_LENGTH = config.FLAG_TARGET_LENGTH
    MODEL_LOAD_PATH = config.MODEL_LOAD_PATH
    BASELINE_LOAD_PATH = config.BASELINE_LOAD_PATH
    BATCH_SIZE = config.BATCH_SIZE
    DEVICE = config.DEVICE

    seed_everything(config.SEED)

    evaluate()
