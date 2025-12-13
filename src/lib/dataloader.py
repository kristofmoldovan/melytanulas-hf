import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F

class FlagDataset(Dataset):
    def __init__(self, csv_file, target_length=1024):
        self.data_frame = pd.read_csv(csv_file)
        self.target_length = target_length
        self.classes = ['Bearish Normal', 'Bearish Pennant', 'Bearish Wedge', 'Bullish Normal', 'Bullish Pennant', 'Bullish Wedge']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load raw data
        file_path = self.data_frame.iloc[idx]['flag_prices_npy_full_path']
        seq_numpy = np.load(file_path).astype(np.float32)

        # Preprocessing ("center vertically", upscale/downscale to fixed width)
        if seq_numpy.max() - seq_numpy.min() > 0:
            norm_seq = (seq_numpy - seq_numpy.min()) / (seq_numpy.max() - seq_numpy.min())
        else:
            norm_seq = seq_numpy - seq_numpy

        seq = torch.from_numpy(norm_seq).view(1, 1, -1)
        seq = F.interpolate(seq, size=self.target_length, mode='linear', align_corners=False).squeeze(0)

        label_str = self.data_frame.iloc[idx]['flag_type']
        label = self.class_to_idx[label_str]

        # RETURN IDX so we can find the raw file for debug purposes!
        return seq, torch.tensor(label, dtype=torch.long), idx