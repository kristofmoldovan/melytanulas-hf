import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

# --- 1. Dataset Class (UNCHANGED) ---
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
        # Safety check: if data segment is shorter than window, return 0
        if len(self.data) <= self.window_size:
            return 0
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        start = idx
        end = idx + self.window_size
        x = self.data[start:end].T # (Features, Window)
        
        targets = []
        # ... (Same FPN Target Generation logic as before) ...
        # (Included in previous full code, abbreviated here for clarity)
        # Note: Ensure you include the full loop logic here when saving
        
        # --- RE-INSERTING THE TARGET LOGIC FOR COMPLETENESS ---
        for config in self.fpn_config:
            stride = config['stride']
            min_len, max_len = config['limit']
            feat_len = self.window_size // stride
            t_cls = torch.zeros(feat_len, dtype=torch.long)
            t_reg = torch.zeros((2, feat_len), dtype=torch.float32)
            t_mask = torch.zeros(feat_len, dtype=torch.float32)
            
            # Optimization: Pre-filter labels? 
            # For simplicity, we stick to the list comp
            window_flags = [l for l in self.labels if l['end'] > start and l['start'] < end]
            
            for flag in window_flags:
                duration = flag['end'] - flag['start']
                if min_len < duration <= max_len:
                    s_rel = max(0, flag['start'] - start)
                    e_rel = min(self.window_size, flag['end'] - start)
                    s_feat = int(s_rel / stride)
                    e_feat = int(e_rel / stride)
                    
                    if e_feat > s_feat:
                        t_cls[s_feat:e_feat] = flag['class']
                        t_mask[s_feat:e_feat] = 1.0
                        for t in range(s_feat, e_feat):
                            t_reg[0, t] = t - (s_rel / stride)
                            t_reg[1, t] = (e_rel / stride) - t
            
            targets.append({'cls': t_cls, 'reg': t_reg, 'mask': t_mask})
        # ------------------------------------------------------
            
        return torch.tensor(x, dtype=torch.float32), targets


# --- 2. The New Splitting Logic ---
def create_time_split_datasets(file_pairs, window_size=512, split_ratios=[0.6, 0.2, 0.2]):
    """
    Splits EACH file chronologically into Train/Val/Test segments.
    Returns: (train_dataset, val_dataset, test_dataset)
    """
    
    # Containers to hold the sub-datasets from all files
    all_train = []
    all_val = []
    all_test = []
    
    class_mapping = {"BullFlag": 1, "BearFlag": 2, "Pennant": 3}

    for price_path, label_path in file_pairs:
        print(f"Processing: {price_path}...")
        
        # --- A. Load & Preprocess File ---
        df_prices = pd.read_csv(price_path)
        if 'price' in df_prices.columns: df_prices.rename(columns={'price': 'close'}, inplace=True)
        
        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp']).dt.tz_localize(None)
        df_prices = df_prices.sort_values('timestamp').reset_index(drop=True)
        
        # Global Time-to-Index Map (for this file)
        # We need this to map the labels to the integer row numbers
        time_to_idx = {t: i for i, t in enumerate(df_prices['timestamp'])}
        
        # Normalize Data
        raw_data = df_prices['close'].values.reshape(-1, 1)
        norm_data = (raw_data - raw_data.mean()) / (raw_data.std() + 1e-6)
        data_array = norm_data.astype(np.float32)
        
        # Load Labels
        df_labels = pd.read_csv(label_path)
        file_labels = []
        for _, row in df_labels.iterrows():
            t_s = pd.to_datetime(row['start']).tz_localize(None)
            t_e = pd.to_datetime(row['end']).tz_localize(None)
            c_id = class_mapping.get(row['flag'])
            
            if c_id and t_s in time_to_idx and t_e in time_to_idx:
                idx_s, idx_e = time_to_idx[t_s], time_to_idx[t_e]
                if idx_e > idx_s:
                    file_labels.append({'start': idx_s, 'end': idx_e, 'class': c_id})

        # --- B. Calculate Split Indices ---
        total_len = len(data_array)
        train_end = int(total_len * split_ratios[0])
        val_end = int(total_len * (split_ratios[0] + split_ratios[1]))
        
        # Define ranges [start, end)
        # We perform "Hard Cuts". 
        # Note: Flags crossing the boundary (e.g. index 5990 to 6010) will be cut or ignored.
        ranges = [
            (0, train_end),          # Train
            (train_end, val_end),    # Val
            (val_end, total_len)     # Test
        ]
        
        # --- C. Create 3 Sub-Datasets for this specific file ---
        # We create independent Dataset objects for each slice.
        # This is safer than Subset() because it physically isolates the data array.
        
        for i, (r_start, r_end) in enumerate(ranges):
            if r_end - r_start <= window_size:
                print(f"  > Segment {i} too short ({r_end-r_start}). Skipping.")
                continue
                
            # 1. Slice the Data Array
            # Use .copy() to ensure memory is contiguous and separate
            data_slice = data_array[r_start:r_end].copy()
            
            # 2. Slice the Labels
            # We must shift label indices because the new slice starts at 0!
            # If a label was at global index 1050, and this slice starts at 1000, 
            # the new label index is 50.
            labels_slice = []
            for l in file_labels:
                # Check if label is fully inside this segment
                if l['start'] >= r_start and l['end'] < r_end:
                    labels_slice.append({
                        'start': l['start'] - r_start, # SHIFT OFFSET
                        'end': l['end'] - r_start,     # SHIFT OFFSET
                        'class': l['class']
                    })
            
            ds = FPN_Dataset(data_slice, labels_slice, window_size)
            
            if i == 0: all_train.append(ds)
            elif i == 1: all_val.append(ds)
            else: all_test.append(ds)

    # --- D. Merge ---
    if not all_train: raise ValueError("No training data found!")
    
    master_train = ConcatDataset(all_train)
    master_val = ConcatDataset(all_val) if all_val else None
    master_test = ConcatDataset(all_test) if all_test else None
    
    print(f"Splitting Complete.")
    print(f"Train: {len(master_train)} samples")
    print(f"Val:   {len(master_val) if master_val else 0} samples")
    print(f"Test:  {len(master_test) if master_test else 0} samples")
    
    return master_train, master_val, master_test