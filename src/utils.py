# Utility functions
# Common helper functions used across the project.
import logging
import sys
import config
import torch
from torch.utils.data import random_split

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
