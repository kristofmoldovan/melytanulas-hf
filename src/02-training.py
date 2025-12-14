# Model training script
# This script defines the model architecture and runs the training loop.
import config
from utils import setup_logger, create_deterministic_splitted_datasets
from lib.training_loop import train_model # training loop in lib/training_loop.py
from lib.baseline_model import BaselineClassifier
from lib.model import FlagClassifier
from lib.dataloader import FlagDataset
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

logger = setup_logger()


def train():
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}")
    
    #
    # Setup data loaders
    #

    #Load and split data

    # A. Load Data
    full_dataset = FlagDataset(csv_file=CSV_FILE, target_length=FLAG_TARGET_LENGTH) #Using the choosen target length
    num_classes = len(full_dataset.classes)
    print(f"Loaded {len(full_dataset)} samples with {num_classes} classes.")

    # B. Split Data (Option 1: Fractional)
    # [0.7, 0.2, 0.1]
    train_ds, val_ds, test_ds = create_deterministic_splitted_datasets(full_dataset)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # C. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


    #
    # Do the training
    #


    # 1. Train the baseline model

    #Resetting seed
    seed_everything(SEED) # <--- Reset state

    baseline_model = BaselineClassifier(num_classes=6) # We use the baseline model here
    baseline_model, baseline_history = train_model(
        baseline_model, train_loader, val_loader,
        epochs=EPOCHS,
        device=DEVICE,
        early_stop_patience=EARLY_STOP_PATIENCE,
        save_path=BASELINE_SAVE_PATH,
        history_save_path=config.BASELINE_TRAINING_HISTORY_PATH
    )

    # 2. Train the hopefully better model

    #Resetting seed
    seed_everything(SEED) # <--- Reset state

    model = FlagClassifier(num_classes=6) # We use the bigger model here
    model, model_history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS,
        device=DEVICE,
        early_stop_patience=EARLY_STOP_PATIENCE,
        save_path=MODEL_SAVE_PATH,
        history_save_path=config.MODEL_TRAINING_HISTORY_PATH
    )
    
    logger.info("Training complete.")

if __name__ == "__main__":
    # Using config values

    CSV_FILE = config.CSV_FILE
    SEED = config.SEED

    DATA_SPLIT_RATIO = config.DATA_SPLIT_RATIO

    BATCH_SIZE = config.BATCH_SIZE
    LEARNING_RATE = config.LEARNING_RATE
    EPOCHS = config.EPOCHS
    DEVICE = config.DEVICE
    EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
    FLAG_TARGET_LENGTH = config.FLAG_TARGET_LENGTH

    # Paths
    DATA_DIR = config.DATA_DIR 
    MODEL_SAVE_PATH = config.MODEL_SAVE_PATH
    BASELINE_SAVE_PATH = config.BASELINE_SAVE_PATH

    seed_everything(SEED)

    #Create output directory if not exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BASELINE_SAVE_PATH), exist_ok=True)

    train()
