import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# --- 1. The Dataset Class (UNCHANGED) ---
# Keep the exact same FPN_Dataset class from the previous answer here.
# It handles the logic for ONE file perfectly.
class FPN_Dataset(Dataset):
    def __init__(self, data, labels, window_size=512):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.fpn_config = [
            {'stride': 1, 'limit': [0, 50]},
            {'stride': 2, 'limit': [50, 150]},
            {'stride': 4, 'limit': [150, 9999]}
        ]

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # (Paste the __getitem__ logic from the previous response here)
        # For brevity, I am not repeating the 50 lines of getitem code 
        # but you MUST include it.
        pass 

# --- 2. The Multi-File Loader Function ---
def create_composite_dataset(file_pairs, window_size=512):
    """
    file_pairs: List of tuples -> [('price1.csv', 'label1.csv'), ('price2.csv', 'label2.csv')]
    """
    datasets = []
    
    # Define mapping: Text Flag -> ID
    class_mapping = {"BullFlag": 1, "BearFlag": 2, "Pennant": 3}

    for price_path, label_path in file_pairs:
        print(f"Processing pair: {price_path} | {label_path}")
        
        # --- A. Load Prices ---
        df_prices = pd.read_csv(price_path)
        # Ensure correct column name 'timestamp' and 'price' (or 'close')
        # If your CSV has 'price', rename it to 'close' for consistency
        if 'price' in df_prices.columns:
            df_prices.rename(columns={'price': 'close'}, inplace=True)
            
        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp']).dt.tz_localize(None)
        df_prices = df_prices.sort_values('timestamp').reset_index(drop=True)
        
        time_to_idx = {t: i for i, t in enumerate(df_prices['timestamp'])}
        
        # Normalize THIS file specifically
        # (Crucial: prevents price differences between File 1 ($100) and File 2 ($50000) from breaking the model)
        raw_data = df_prices['close'].values.reshape(-1, 1)
        norm_data = (raw_data - raw_data.mean()) / (raw_data.std() + 1e-6)
        data_array = norm_data.astype(np.float32)

        # --- B. Load Labels ---
        df_labels = pd.read_csv(label_path)
        file_labels = []
        
        for _, row in df_labels.iterrows():
            # Using your column names: flag, start, end
            t_start = pd.to_datetime(row['start']).tz_localize(None)
            t_end = pd.to_datetime(row['end']).tz_localize(None)
            flag_name = row['flag']
            
            c_id = class_mapping.get(flag_name)
            if c_id is None: continue # Skip unknown flags
            
            idx_start = time_to_idx.get(t_start)
            idx_end = time_to_idx.get(t_end)
            
            if idx_start is not None and idx_end is not None and idx_end > idx_start:
                file_labels.append({'start': idx_start, 'end': idx_end, 'class': c_id})
        
        # --- C. Create Dataset for this file ---
        # Only create if we actually have data
        if len(data_array) > window_size:
            ds = FPN_Dataset(data_array, file_labels, window_size=window_size)
            datasets.append(ds)
        else:
            print(f"Warning: {price_path} is too short for window size {window_size}. Skipping.")

    # --- D. Merge into One ---
    if not datasets:
        raise ValueError("No valid datasets were created!")
        
    master_dataset = ConcatDataset(datasets)
    print(f"Success! Combined {len(datasets)} files into one dataset.")
    return master_dataset