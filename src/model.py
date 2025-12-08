import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FPN_ActionFormer(nn.Module):
    def __init__(self, in_channels, num_classes, emb_dim=256):
        super().__init__()
        self.num_classes = num_classes
        
        # --- 1. Stem (Input Projection) ---
        self.stem = nn.Conv1d(in_channels, emb_dim, kernel_size=3, padding=1)
        
        # --- 2. Transformer Backbone (Multi-Scale) ---
        # Level 1: High Resolution (Stride 1) - Detects Small Flags
        self.layer1_enc = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.layer1_block = nn.TransformerEncoder(self.layer1_enc, num_layers=2)
        
        # Downsample 1 -> 2 (Stride 2)
        self.down1 = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, stride=2, padding=1)
        
        # Level 2: Medium Resolution (Stride 2) - Detects Medium Flags
        self.layer2_enc = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.layer2_block = nn.TransformerEncoder(self.layer2_enc, num_layers=2)
        
        # Downsample 2 -> 3 (Stride 2)
        self.down2 = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, stride=2, padding=1)
        
        # Level 3: Low Resolution (Stride 4) - Detects Large Flags
        self.layer3_enc = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.layer3_block = nn.TransformerEncoder(self.layer3_enc, num_layers=2)

        # --- 3. Feature Pyramid Network (FPN) ---
        # Lateral connections to smooth features
        self.lat_p3 = nn.Conv1d(emb_dim, emb_dim, 1)
        self.lat_p2 = nn.Conv1d(emb_dim, emb_dim, 1)
        self.lat_p1 = nn.Conv1d(emb_dim, emb_dim, 1)
        
        # --- 4. Prediction Heads (Shared Weights) ---
        # These heads slide over ALL pyramid levels
        self.cls_head = nn.Conv1d(emb_dim, num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(emb_dim, 2, kernel_size=3, padding=1) 

        # Init weights (Important for FPN stability)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize regression head to output small values initially
        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.constant_(self.reg_head.bias, 0)

    def forward(self, x):
        # x: (Batch, C, Time)
        
        # --- Bottom-Up Path (Transformer Encoding) ---
        # L1
        x1 = self.stem(x)
        x1_t = x1.permute(0, 2, 1) # (B, T, C)
        l1 = self.layer1_block(x1_t).permute(0, 2, 1) # (B, C, T)
        
        # L2
        x2 = self.down1(l1)
        x2_t = x2.permute(0, 2, 1)
        l2 = self.layer2_block(x2_t).permute(0, 2, 1)
        
        # L3
        x3 = self.down2(l2)
        x3_t = x3.permute(0, 2, 1)
        l3 = self.layer3_block(x3_t).permute(0, 2, 1)
        
        # --- Top-Down Path (FPN) ---
        p3 = self.lat_p3(l3)
        
        # P3 upsampled + P2
        p2 = self.lat_p2(l2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        
        # P2 upsampled + P1
        p1 = self.lat_p1(l1) + F.interpolate(p2, scale_factor=2, mode='nearest')
        
        features = [p1, p2, p3] # Stride 1, 2, 4
        
        # --- Heads ---
        cls_preds = []
        reg_preds = []
        
        for feat in features:
            cls_preds.append(self.cls_head(feat))
            # ReLU because distance cannot be negative
            reg_preds.append(F.relu(self.reg_head(feat)))
            
        return cls_preds, reg_preds