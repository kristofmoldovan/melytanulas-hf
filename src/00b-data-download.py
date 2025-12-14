# Script to download and setup raw data from Google Drive
import os
import sys
import shutil
import zipfile
import gdown

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils import setup_logger

logger = setup_logger()

# ==========================================
# CONFIGURATION
# ==========================================
# Paste your Google Drive File ID here.
# Link format: https://drive.google.com/file/d/YOUR_ID_HERE/view?usp=sharing
GDRIVE_FILE_ID = "13ZnY-VHgNJzQuTGrZ-5fhXBPjfbV1dqA" 
# ==========================================

def download_file_from_google_drive(id, destination):
    url = f'https://drive.google.com/uc?id={id}'
    gdown.download(url, destination, quiet=False)

def extract_and_setup_data(zip_path, target_root):
    temp_extract_dir = "temp_extract_bullflag"
    
    # 1. Extract Zip
    logger.info(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
    except zipfile.BadZipFile:
        logger.error("Downloaded file is not a valid zip file.")
        return

    # 2. Locate specific folder
    source_folder_name = "bullflagdetector"
    source_path = os.path.join(temp_extract_dir, source_folder_name)
    
    if not os.path.exists(source_path):
        logger.error(f"Folder '{source_folder_name}' not found in the zip archive.")
        logger.info(f"Contents: {os.listdir(temp_extract_dir)}")
        shutil.rmtree(temp_extract_dir)
        return

    # 3. Move contents to target
    logger.info(f"Moving contents to {target_root}...")
    os.makedirs(target_root, exist_ok=True)

    items = os.listdir(source_path)
    for item in items:
        s = os.path.join(source_path, item)
        d = os.path.join(target_root, item)
        
        if os.path.exists(d):
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
        
        shutil.move(s, d)
        logger.info(f"Moved: {item}")

    # 4. Cleanup
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(temp_extract_dir)
    os.remove(zip_path)
    logger.info("Data setup complete.")

if __name__ == "__main__":
    if GDRIVE_FILE_ID == "YOUR_GDRIVE_FILE_ID_HERE":
        logger.warning("Please set the 'GDRIVE_FILE_ID' variable in this script before running.")
        sys.exit(1)

    zip_filename = "data_download.zip"
    target_dir = config.STARTING_FOLDER # /app/data/raw

    try:
        download_file_from_google_drive(GDRIVE_FILE_ID, zip_filename)
        extract_and_setup_data(zip_filename, target_dir)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        # Cleanup
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        if os.path.exists("temp_extract_bullflag"):
            shutil.rmtree("temp_extract_bullflag")