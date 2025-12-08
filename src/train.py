import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import random
import numpy as np

# --- Imports from your files ---
from model import FPN_ActionFormer
from data_loader import create_time_split_datasets

# --- Configuration ---
CONFIG = {
    'seed': 42,            # Ensures exact reproducibility
    'window_size': 512,    # Must match your data_loader
    'batch_size': 16,
    'lr': 1e-4,            # Learning rate (lower is usually safer for Transformers)
    'epochs': 50,
    'num_classes': 3,      # 1=Bull, 2=Bear, 3=Pennant
    'in_channels': 1,      # 1 because we use Close price only
    'save_dir': 'checkpoints',
    'files': [
        # Pair 1: (Price CSV, Label CSV)
        #('data/BTC_prices.csv', 'data/BTC_labels.csv'),
        # Add more pairs here...
        # ('data/ETH_prices.csv', 'data/ETH_labels.csv'),
        ('data/data_EURUSD_1H_005.csv', 'data/labels_EURUSD_1H_005.csv'),
    ]
}

# --- 1. Helper: Set Seed ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. Loss Function (Focal + L1) ---
class ActionFormerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_loss = nn.L1Loss(reduction='none')

    def sigmoid_focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """
        Focal Loss addresses the class imbalance (most points are background).
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()

    def forward(self, pred_cls_list, pred_reg_list, targets):
        total_cls_loss = 0
        total_reg_loss = 0
        
        # Loop over the 3 levels of the Feature Pyramid
        for i in range(3):
            # Predictions
            p_cls = pred_cls_list[i] # (Batch, Classes, Time)
            p_reg = pred_reg_list[i] # (Batch, 2, Time)
            
            # Targets (Collated from the batch list)
            t_cls = torch.stack([t[i]['cls'] for t in targets]).to(p_cls.device)
            t_reg = torch.stack([t[i]['reg'] for t in targets]).to(p_reg.device)
            t_mask = torch.stack([t[i]['mask'] for t in targets]).to(p_reg.device)

            # --- Classification Loss ---
            # Prepare One-Hot targets for Focal Loss
            # Input t_cls is (Batch, Time) -> Output needs (Batch, Classes, Time)
            t_cls_onehot = F.one_hot(t_cls, num_classes=p_cls.shape[1] + 1).float()
            
            # Slice off class 0 (background) and permute to (B, C, T)
            t_cls_onehot = t_cls_onehot[:, :, 1:].permute(0, 2, 1)
            
            total_cls_loss += self.sigmoid_focal_loss(p_cls, t_cls_onehot)

            # --- Regression Loss ---
            # Only punish regression errors where there is actually an object (mask=1)
            reg_loss_raw = self.reg_loss(p_reg, t_reg)
            num_positives = t_mask.sum() + 1e-6 # Avoid div by zero
            
            # Sum errors, mask them, normalize by number of flags
            total_reg_loss += (reg_loss_raw.sum(dim=1) * t_mask).sum() / num_positives

        # Weighted sum (Regression is usually harder, so sometimes given higher weight)
        return total_cls_loss + total_reg_loss

# --- 3. Main Training Loop ---
def train():
    # Setup
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # A. Load Datasets
    print("Loading and Splitting Data...")
    # This function creates 3 separate datasets (Time-Split)
    train_ds, val_ds, test_ds = create_time_split_datasets(
        CONFIG['files'], 
        window_size=CONFIG['window_size'],
        split_ratios=[0.6, 0.2, 0.2] # 60% Train, 20% Val, 20% Test
    )
    
    # We ignore test_ds for now (it will be used in test.py)
    print(f"Dataset Sizes -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # B. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    # C. Initialize Model
    model = FPN_ActionFormer(
        in_channels=CONFIG['in_channels'], 
        num_classes=CONFIG['num_classes']
    ).to(device)

    # D. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    criterion = ActionFormerLoss()

    best_val_loss = float('inf')
    print("Starting Training...")

    # E. Epoch Loop
    for epoch in range(CONFIG['epochs']):
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0
        
        for batch_idx, (x, targets) in enumerate(train_loader):
            x = x.to(device)
            
            # Forward
            pred_cls, pred_reg = model(x)
            
            # Loss
            loss = criterion(pred_cls, pred_reg, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (Prevents exploding gradients in Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"  > Epoch {epoch+1} Step {batch_idx} Loss: {loss.item():.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(device)
                pred_cls, pred_reg = model(x)
                loss = criterion(pred_cls, pred_reg, targets)
                val_loss += loss.item()

        # Averages
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # --- SAVE BEST MODEL ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(CONFIG['save_dir'], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Improved! Saved best model to {save_path}")
            
    print("Training Complete.")

if __name__ == "__main__":
    train()