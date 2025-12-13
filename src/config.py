import torch
import os
from datetime import datetime

# Configuration settings as an example


# --- Seed ---
SEED = 42

# --- Preprocessing Configuration ---

STARTING_FOLDER = '/app/data/raw' # The main folder containing child folders (NEPTUN codes)
OUTPUT_ROOT_PATH = "/app/data/processed"



# --- CONFIGURATION FOR TRAINING ---

USE_CACHED_DATASET = True # If True, uses cached dataset in memory for faster training (uses ~700MB of RAM)

DATA_SPLIT_RATIO = [0.7, 0.2, 0.1] # Train, Val, Test split ratios, must add up to 1.0, and 3 elements!

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 200 # 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOP_PATIENCE = 10
FLAG_TARGET_LENGTH = 512


#
RUN_ID = "1" # Change if you want to keep earlier trained models (otherwise the the same path is used and overwritten)
# Paths
DATA_DIR = OUTPUT_ROOT_PATH # Can be modified if needed, but structure must be the same
MODEL_SAVE_PATH = f"/app/trained_models/{RUN_ID}/model.pth"
BASELINE_SAVE_PATH = f"/app/trained_models/{RUN_ID}/baseline_model.pth"




# --- CONFIGURATION FOR EVALUATION & INFERENCE ---

MODEL_LOAD_PATH = MODEL_SAVE_PATH # Path to the trained model file, you can use pretrained models here too
BASELINE_LOAD_PATH = BASELINE_SAVE_PATH # Path to the baseline model file, you can use pretrained models here too




# --- Dont change these ---
TRIM_N = 9 # Number of characters to trim from the start of filenames (needed because of labeling app export format)
CSV_FILE = os.path.join(OUTPUT_ROOT_PATH, 'consolidated_labels.csv')  # Path to the consolidated CSV file (it contains the labels and their data)

