#Define the training loop

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, 
                epochs=50, lr=0.001, device='cpu', early_stop_patience=10, save_path="best_model.pth"):
    """
    Trains a PyTorch model with Early Stopping and LR Scheduling.
    
    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs (int): Maximum number of epochs.
        lr (float): Initial learning rate.
        device (str): 'cuda' or 'cpu'.
        early_stop_patience (int): Epochs to wait for improvement before stopping.
        save_path (str): Path to save the best model weights.
    Returns:
        model: The model with the BEST weights loaded.
        history: Dictionary containing loss and accuracy curves.
    """
    
    # 1. Setup
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler: Reduce LR by half if val_loss doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    early_stop_counter = 0

    print(f"\n--- Starting Training {os.path.basename(save_path)}---")
    start_time = time.time()

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # --- Logging ---
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Step Scheduler
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:02d} | LR: {current_lr:.1e} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2%} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2%}")

        # --- Early Stopping & Saving ---
        if epoch_val_loss < (best_val_loss - 0.001):
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"\n--- Early Stopping Triggered at Epoch {epoch+1} ---")
            break

    # Final Cleanup
    duration = time.time() - start_time
    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}. Time: {duration:.0f}s")
    
    # Load best weights so the returned model is optimized
    model.load_state_dict(torch.load(save_path))
    
    return model, history