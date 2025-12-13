import torch
import os

# Configuration settings as an example


# --- Seed ---
SEED = 42

# --- Preprocessing Configuration ---

STARTING_FOLDER = '/app/data/raw' # The main folder containing child folders (NEPTUN codes)
OUTPUT_ROOT_PATH = "/app/data/processed"

# --- CONFIGURATION FOR TRAINING ---

DATA_SPLIT_RATIO = [0.7, 0.2, 0.1] # Train, Val, Test split ratios, must add up to 1.0, and 3 elements!

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOP_PATIENCE = 10
FLAG_TARGET_LENGTH = 512

# Paths
DATA_DIR = OUTPUT_ROOT_PATH # Can be modified if needed, but structure must be the same
MODEL_SAVE_PATH = "/app/model.pth"
BASELINE_SAVE_PATH = "/app/baseline_model.pth"

# --- Dont change these ---
TRIM_N = 9 # Number of characters to trim from the start of filenames (needed because of labeling app export format)
CSV_FILE = os.path.join(OUTPUT_ROOT_PATH, 'consolidated_labels.csv')  # Path to the consolidated CSV file (it contains the labels and their data)