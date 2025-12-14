# Script to print configuration and model details
# Useful for verifying settings and model architectures before training.

import config
import sys
import os
import torch

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import setup_logger
from lib.baseline_model import BaselineClassifier
from lib.model import FlagClassifier

logger = setup_logger()

def explain_config_var(var_name):
    """Returns a short explanation for known configuration variables."""
    explanations = {
        "EPOCHS": "Total number of training iterations over the entire dataset.",
        "BATCH_SIZE": "Number of samples processed before the model is updated.",
        "LEARNING_RATE": "Step size for the optimizer during gradient descent.",
        "DATA_SPLIT_RATIO": "Distribution ratio for Train, Validation, and Test sets.",
        "SEED": "Random seed for reproducibility across runs.",
        "DEVICE": "Computation device (CPU or CUDA) used for training.",
        "EARLY_STOP_PATIENCE": "Epochs to wait for validation loss improvement before stopping.",
        "FLAG_TARGET_LENGTH": "Fixed input sequence length (time steps) for the model.",
        "CSV_FILE": "Path to the processed metadata CSV file.",
        "DATA_DIR": "Root directory for dataset storage.",
        "MODEL_SAVE_PATH": "Destination path for saving the best FlagClassifier weights.",
        "BASELINE_SAVE_PATH": "Destination path for saving the best BaselineClassifier weights.",
        "STARTING_FOLDER": "Source folder containing raw JSON/CSV data.",
        "OUTPUT_ROOT_PATH": "Target folder for processed data artifacts.",
        "TRIM_N": "Character count to trim from filenames during preprocessing.",
        "HEADLESS_PLOT": "Toggle for running Matplotlib without a GUI.",
        "PLOT_OUTPUT_FOLDER": "Directory for saving generated plots.",
        "BASELINE_TRAINING_HISTORY_PATH": "CSV path for baseline training metrics.",
        "MODEL_TRAINING_HISTORY_PATH": "CSV path for main model training metrics."
    }
    return explanations.get(var_name, "Configuration setting.")

def print_config_variables():
    logger.info("========================================")
    logger.info("       CONFIGURATION VARIABLES")
    logger.info("========================================")
    
    # Get all attributes of config module that don't start with __
    config_vars = [attr for attr in dir(config) if not attr.startswith("__")]
    
    for var in config_vars:
        value = getattr(config, var)
        # Filter out modules or functions, keep simple types
        if isinstance(value, (int, float, str, list, dict, bool, tuple)):
            explanation = explain_config_var(var)
            logger.info(f"{var}: {value}")
            logger.info(f"  -> {explanation}\n")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info():
    logger.info("========================================")
    logger.info("       MODEL ARCHITECTURES")
    logger.info("========================================")
    
    # Define models to inspect
    # Assuming 6 classes as used in 02-training.py
    models_to_inspect = [
        ("BaselineClassifier", BaselineClassifier(num_classes=6), 
         "Reference model. A shallow 1D-CNN architecture designed to establish a performance baseline. It uses fewer convolutional layers and filters compared to the main model."),
        ("FlagClassifier", FlagClassifier(num_classes=6), 
         "Main project model. A deeper 1D-CNN architecture with more convolutional layers and filters. It is designed to capture more complex, hierarchical temporal features in the price data.")
    ]

    for name, model, description in models_to_inspect:
        params = count_parameters(model)
        logger.info(f"--- {name} ---")
        logger.info(f"Total Trainable Parameters: {params:,}")
        logger.info(f"Description: {description}")
        logger.info(f"Architecture:\n{model}\n")

if __name__ == "__main__":
    print_config_variables()
    print_model_info()
