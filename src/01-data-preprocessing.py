# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
from utils import setup_logger
import os
import glob
import json
import pandas as pd
import numpy as np
import config

logger = setup_logger()

def clean_timestamps_vectorized(series):
    """
    Vectorized function to handle mixed timestamp formats (Unix str & Date str).
    Returns integer timestamps in SECONDS (Unix Epoch).
    """
    series = series.astype(str)
    numeric = pd.to_numeric(series, errors='coerce')
    mask_dates = numeric.isna()

    if mask_dates.any():
        dates = pd.to_datetime(series[mask_dates], errors='coerce')
        # Convert to SECONDS (Unix Standard)
        numeric.loc[mask_dates] = dates.astype(np.int64) // 10**9

    return numeric.fillna(0).astype('int64')

def process_subfolder(folder_path, cwd):
    """
    Processes a single subfolder.
    Returns a list of records (dicts) for the master CSV.
    """
    records = []
    folder_name = os.path.basename(folder_path)
    print(f"\n--- Processing Subfolder: {folder_name} ---")

    json_path_pattern = os.path.join(folder_path, "*.json")
    json_files = glob.glob(json_path_pattern)

    if not json_files:
        print(f"Skipping {folder_name}: No JSON file found.")
        return []

    json_file_path = json_files[0]

    # Create the 'processed_data' folder inside the subfolder
    output_path = os.path.join(OUTPUT_ROOT_PATH, folder_name)
    os.makedirs(output_path, exist_ok=True)

    try:
        with open(json_file_path, 'r') as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"Error loading JSON in {folder_name}: {e}")
        return []

    for entry in data_list:
        json_filename = entry.get('file_upload')

        # Clean filename logic (removing prefixes)
        if json_filename and len(json_filename) > TRIM_N:
            real_csv_name = json_filename[TRIM_N:]
        else:
            real_csv_name = json_filename

        input_csv_path = os.path.join(folder_path, real_csv_name)

        if not real_csv_name or not os.path.exists(input_csv_path):
            print(f"  [!] Missing file: {real_csv_name}")
            continue

        print(f"  Processing: {real_csv_name}")

        # ---------------------------------------------------------
        # STEP 1: Process Main Data CSV
        # ---------------------------------------------------------
        try:
            df = pd.read_csv(input_csv_path, dtype={'timestamp': str})

            # --- CHANGE: specific check for "Close" vs "close" ---
            if 'Close' in df.columns and 'close' not in df.columns:
                df.rename(columns={'Close': 'close'}, inplace=True)

            if 'timestamp' in df.columns and 'close' in df.columns:
                df['timestamp'] = clean_timestamps_vectorized(df['timestamp'])
                df.sort_values('timestamp', inplace=True)
            else:
                print(f"    -> Warning: File skipped, columns missing in {real_csv_name} (Found: {list(df.columns)})")
                continue
        except Exception as e:
            print(f"    -> Error reading CSV: {e}")
            continue

        # ---------------------------------------------------------
        # STEP 2: Extract Labels
        # ---------------------------------------------------------
        raw_labels = []
        for annotation in entry.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('type') == 'timeserieslabels':
                    value = result.get('value', {})
                    ts_labels = value.get('timeserieslabels', [])
                    label_name = ts_labels[0] if ts_labels else None

                    if label_name:
                        raw_labels.append({
                            'flag_type': label_name,
                            'start': value.get('start'),
                            'end': value.get('end')
                        })

        if not raw_labels:
            print(f"    -> No labels found for {real_csv_name}")
            continue

        df_labels = pd.DataFrame(raw_labels)
        df_labels['start'] = clean_timestamps_vectorized(df_labels['start'])
        df_labels['end'] = clean_timestamps_vectorized(df_labels['end'])

        # ---------------------------------------------------------
        # STEP 3: Create Slices & Accumulate Metadata
        # ---------------------------------------------------------
        for _, row in df_labels.iterrows():
            label_name = row['flag_type']
            start_ts = row['start']
            end_ts = row['end']

            # Create Class Folder (inside the OUTPUT_ROOT_FOLDER processed_data)
            class_dir = os.path.join(output_path, label_name)
            os.makedirs(class_dir, exist_ok=True)

            # Filter Main Data
            mask_slice = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
            df_slice = df.loc[mask_slice, ['timestamp', 'close']]

            if not df_slice.empty:
                # Remove '.csv' extension from the input filename
                clean_name_base = real_csv_name.replace('.csv', '')

                # Construct base name
                base_name = f"{clean_name_base}_{start_ts}_{end_ts}"

                csv_filename = base_name + ".csv"
                npy_filename = base_name + ".npy"

                # Create Absolute Full Paths
                csv_full_path = os.path.abspath(os.path.join(class_dir, csv_filename))
                npy_full_path = os.path.abspath(os.path.join(class_dir, npy_filename))

                # Save Data
                df_slice.to_csv(csv_full_path, index=False)

                prices_array = df_slice['close'].values
                if len(prices_array) != len(df_slice):
                    raise ValueError("Length mismatch between CSV and NPY array")
                np.save(npy_full_path, prices_array)

                # Add to Local Record List
                records.append({
                    "processed_folder_name": folder_name, # New Column
                    "input_csv_file": real_csv_name,
                    "flag_type": label_name,
                    "start": start_ts,
                    "end": end_ts,
                    "flag_csv_full_path": csv_full_path,
                    "flag_prices_npy_full_path": npy_full_path,
                    "flag_length": len(prices_array)
                })

        print(f"    -> Slices generated.")

    return records


def process_root_data_folder():
    # 1. Capture the absolute path of your current working directory
    cwd = os.path.abspath(os.getcwd())
    print(f"DEBUG: Script running from CWD: {cwd}")

    if not os.path.exists(STARTING_FOLDER):
        print(f"Error: Starting folder '{STARTING_FOLDER}' does not exist.")
        return

    all_consolidated_records = []

    # --- NEW: Define subfolders to skip ---
    SUBFOLDERS_TO_SKIP = ['TYEGJ8','VWXUD6', 'consensus', 'sample'] # Add folder names here

    # 2. Iterate through all items in the STARTING_FOLDER
    # --- CHANGE: Wrap os.listdir in sorted() to ensure order ---
    items = sorted(os.listdir(STARTING_FOLDER))

    # Filter only directories
    subfolders = [os.path.join(STARTING_FOLDER, item) for item in items if os.path.isdir(os.path.join(STARTING_FOLDER, item))]

    print(f"Found {len(subfolders)} subfolders to process.")

    # 3. Process each subfolder
    for subfolder_path in subfolders:
        subfolder_name = os.path.basename(subfolder_path)
        if subfolder_name in SUBFOLDERS_TO_SKIP:
            print(f"Skipping subfolder: {subfolder_name} (as requested).")
            continue

        subfolder_records = process_subfolder(subfolder_path, cwd)
        all_consolidated_records.extend(subfolder_records)

    # 4. Save the SINGLE Master Consolidated CSV in the OUTPUT_ROOT_PATH
    if all_consolidated_records:
        master_df = pd.DataFrame(all_consolidated_records)

        master_csv_path = os.path.join(OUTPUT_ROOT_PATH, "consolidated_labels.csv")

        cols = [
            "processed_folder_name",
            "input_csv_file",
            "flag_type",
            "start",
            "end",
            "flag_csv_full_path",
            "flag_prices_npy_full_path",
            "flag_length"
        ]
        master_df = master_df[cols]

        master_df.to_csv(master_csv_path, index=False)
        print(f"\n==================================================")
        print(f"[Success] Master consolidated file saved at:\n{master_csv_path}")
        print(f"Total records: {len(master_df)}")
        print(f"==================================================")
        return master_df
    else:
        print("\n[Info] No valid labels processed in any folder.")


if __name__ == "__main__":
    logger.info("Preprocessing data...")
    STARTING_FOLDER = config.STARTING_FOLDER
    OUTPUT_ROOT_PATH = config.OUTPUT_ROOT_PATH
    TRIM_N = config.TRIM_N

    os.makedirs(OUTPUT_ROOT_PATH, exist_ok=True)

    df = process_root_data_folder()
    df.head()
    
    logger.info("Data preprocessing complete.")

