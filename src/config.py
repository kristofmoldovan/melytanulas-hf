import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt



# --- Seed ---
SEED = 42

# --- Preprocessing Configuration ---

STARTING_FOLDER = '/app/data/raw' # The main folder containing child folders (NEPTUN codes)
OUTPUT_ROOT_PATH = "/app/data/processed" # Root folder for processed data output



# --- CONFIGURATION FOR TRAINING ---

USE_CACHED_DATASET = True # If True, uses cached dataset in memory for faster training (uses ~700MB of RAM)

DATA_SPLIT_RATIO = [0.7, 0.2, 0.1] # Train, Val, Test split ratios, must add up to 1.0, and 3 elements!

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 200 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOP_PATIENCE = 10
FLAG_TARGET_LENGTH = 512 # Model input length after interpolation, fixed for model


RUN_ID = "latest" # Change if you want to keep earlier trained models (used in path for output, which is overwritten)
# Paths
DATA_DIR = OUTPUT_ROOT_PATH # Can be modified if needed, but structure must be the same
MODEL_SAVE_PATH = f"/app/output/{RUN_ID}/model.pth" # Path to save model
BASELINE_SAVE_PATH = f"/app/output/{RUN_ID}/baseline_model.pth" # Path to save baseline model

MODEL_TRAINING_HISTORY_PATH = f"/app/output/{RUN_ID}/model_history.csv" # Path to save model training history CSV
BASELINE_TRAINING_HISTORY_PATH = f"/app/output/{RUN_ID}/baseline_history.csv" # Path to save baseline training history CSV



HEADLESS_PLOT = True # If True, saves plots to disk instead of opening a window
PLOT_OUTPUT_FOLDER = f"/app/output/{RUN_ID}/plots" # Where to save training/validation plots, only used if HEADLESS_PLOT is True



# --- CONFIGURATION FOR EVALUATION & INFERENCE ---

MODEL_LOAD_PATH = MODEL_SAVE_PATH # Path to the trained model file, you can use pretrained models here too if you want to skip training
BASELINE_LOAD_PATH = BASELINE_SAVE_PATH # Path to the baseline model file, you can use pretrained models here too if you want to skip training


# --- CONFIGURATION FOR PREDICTION IN INFERENCE  ---

PREDICT_INPUT_CSV = '/app/inference/EURUSD_5min_002.csv'  # Path to the CSV file containing data to predict on
PREDICT_OUTPUT_CSV = f"/app/output/{RUN_ID}/inference_results.csv"  # Path to save inference results CSV
WINDOW_SIZE = 512          # Will be interpolated to model input size, so you can search for any size of flags
STRIDE = 10                 # Moving window stride
CONFIDENCE_THRESHOLD = 0.55 # Only classify if softmax prob > 70%



# --- Dont change these ---
CLASS_LABELS = ['Bearish Normal', 'Bearish Pennant', 'Bearish Wedge', 'Bullish Normal', 'Bullish Pennant', 'Bullish Wedge']
TRIM_N = 9 # Number of characters to trim from the start of filenames (needed because of labeling app export format)
CSV_FILE = os.path.join(OUTPUT_ROOT_PATH, 'consolidated_labels.csv')  # Path to the consolidated CSV file (it contains the labels and their data)

