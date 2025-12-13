import torch
import torch.nn as nn
import torch.optim as optim
import time

class FlagClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1: Conv -> BN -> ReLU -> Pool
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),  # <--- The magic fix
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Layer 2: Conv -> BN -> ReLU
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),  # <--- The magic fix
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)