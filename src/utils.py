# Utility functions
# Common helper functions used across the project.
import logging
import sys
import config
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import matplotlib


def setup_logger(name=__name__):
    """
    Sets up a logger that outputs to the console (stdout).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

#Used for creating the same train/val/test splits in training and evaluation
def create_deterministic_splitted_datasets(full_dataset):
     return random_split(full_dataset, config.DATA_SPLIT_RATIO, generator=torch.Generator().manual_seed(config.SEED))

def load_config():
    pass

import os
import time

logger = setup_logger()
def use_headless_plotting(save_dir):
    """
    Configures Matplotlib to run in headless mode (no window) and 
    redirects plt.show() to save files with unique, timestamped names.
    """
    
    # 1. Force "Headless" mode (prevents crashing on servers/terminals)
    # Must be done before importing pyplot generally, or immediately after
    matplotlib.use('Agg') 
    
    # 2. Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # 3. Define the custom save function (Closure captures 'save_dir')
    def custom_show():
        # Initialize logger (assuming this function exists in your scope)
        # logger = setup_logger() 
        
        # Base settings
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_name = os.path.join(save_dir, f"plot_{timestamp}")
        extension = ".png"
        
        # --- DUPLICATE PROOF LOGIC ---
        # Start with the standard filename
        filename = f"{base_name}{extension}"
        
        # If it exists, loop and append an index (_1, _2, etc.)
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_name}_{counter}{extension}"
            counter += 1
        # -----------------------------

        # Save and Close
        plt.savefig(filename, bbox_inches='tight')
        
        # Print/Log result
        logger.info(f"[Plotting] Plot saved to: {filename}")
        # if logger: logger.info(f"Plot saved to: {filename}")
        
        # Close to free memory
        plt.close()

    # 4. OVERWRITE the default plt.show
    plt.show = custom_show
    logger.info(f"[System] Headless plotting enabled. Plots will save to: {save_dir}")

## Global setup

if (config.HEADLESS_PLOT):
    use_headless_plotting(config.PLOT_OUTPUT_FOLDER)
