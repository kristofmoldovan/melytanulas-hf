# Model training script
# This script defines the model architecture and runs the training loop.
import config
from utils import setup_logger
from lib.training_loop import train_model # training loop in lib/training_loop.py
from lib.baseline_model import BaselineClassifier
from lib.model import FlagClassifier
from lib.dataloader import FlagDataset

logger = setup_logger()


def train():
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}")
    
    #
    # Setup data loaders
    #

    #Load and split data

    # A. Load Data
    full_dataset = FlagDataset(csv_file=CSV_FILE, target_length=FLAG_TARGET_LENGTH) #Using the choosen target length
    num_classes = len(full_dataset.classes)
    print(f"Loaded {len(full_dataset)} samples with {num_classes} classes.")

    # B. Split Data (Option 1: Fractional)
    # [0.7, 0.2, 0.1]
    train_ds, val_ds, test_ds = random_split(full_dataset, DATA_SPLIT_RATIO, generator=torch.Generator().manual_seed(SEED))

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # C. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


    #
    # Do the training
    #


    # 1. Train the baseline model

    #Resetting seed
    seed_everything(42) # <--- Reset state

    baseline_model = BaselineClassifier(num_classes=6)
    baseline_model, baseline_history = train_model(
        baseline_model, train_loader, val_loader,
        tag="baseline",
        epochs=EPOCHS,
        device=DEVICE,
        early_stop_patience=EARLY_STOP_PATIENCE
    )

    # 2. Train the hopefully better model

    #Resetting seed
    seed_everything(42) # <--- Reset state

    model = FlagClassifier(num_classes=6)
    model, model_history = train_model(
        model, train_loader, val_loader,
        tag="FlagClassifier",
        epochs=EPOCHS,
        device=DEVICE,
        early_stop_patience=EARLY_STOP_PATIENCE
    )
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
