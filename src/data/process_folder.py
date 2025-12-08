import json
import pandas as pd
import os
import glob

# Number of characters to trim from the start of the filename in the JSON
# Example: "351f4d2a-EURUSD_1H_005.csv" -> trims 9 chars -> "EURUSD_1H_005.csv"
TRIM_N = 9 

def process_folder():
    # 1. Find the JSON file in the current directory
    json_files = glob.glob("*.json")
    
    if not json_files:
        print("No JSON file found in the current directory.")
        return

    # Use the first JSON file found
    json_file_path = json_files[0]
    print(f"Reading JSON file: {json_file_path}")

    with open(json_file_path, 'r') as f:
        data_list = json.load(f)

    # 2. Iterate through each entry in the JSON
    for entry in data_list:
        json_filename = entry.get('file_upload')
        
        # --- Logic to fix the filename ---
        # Trim the first N characters to get the real CSV name
        if json_filename and len(json_filename) > TRIM_N:
            real_csv_name = json_filename[TRIM_N:]
        else:
            real_csv_name = json_filename

        # Check if the file exists
        if not real_csv_name or not os.path.exists(real_csv_name):
            print(f"Skipping: {real_csv_name} (File not found)")
            continue

        print(f"Processing: {real_csv_name}...")

        # --- Task 1: Create Data CSV (timestamp, close) ---
        try:
            df = pd.read_csv(real_csv_name)
            
            # Check for required columns
            if 'timestamp' in df.columns and 'close' in df.columns:
                df_data = df[['timestamp', 'close']]
                
                # Create a new filename for the data output
                # Example: data_EURUSD_1H_005.csv
                data_csv_name = f"data_{real_csv_name}"
                df_data.to_csv(data_csv_name, index=False)
                print(f"  -> Created: {data_csv_name}")
            else:
                print(f"  -> Warning: 'timestamp' or 'close' column missing in {real_csv_name}")
        except Exception as e:
            print(f"  -> Error reading CSV: {e}")

        # --- Task 2: Create Label CSV (flag_type, start, end) ---
        labels = []
        
        # Loop through annotations and results
        for annotation in entry.get('annotations', []):
            for result in annotation.get('result', []):
                if result.get('type') == 'timeserieslabels':
                    value = result.get('value', {})
                    
                    # Extract the label name (it is a list in the JSON)
                    ts_labels = value.get('timeserieslabels', [])
                    label_name = ts_labels[0] if ts_labels else None
                    
                    if label_name:
                        labels.append({
                            'flag_type': label_name,
                            'start': value.get('start'),
                            'end': value.get('end')
                        })
        
        # Save labels to CSV
        if labels:
            df_labels = pd.DataFrame(labels)
            # Example: labels_EURUSD_1H_005.csv
            label_csv_name = f"labels_{real_csv_name}"
            df_labels.to_csv(label_csv_name, index=False)
            print(f"  -> Created: {label_csv_name}")
        else:
            print(f"  -> No labels found for {real_csv_name}")

if __name__ == "__main__":
    process_folder()