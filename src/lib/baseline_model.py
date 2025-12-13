import torch
import torch.nn as nn
import torch.optim as optim
import time

#
# ~ 200 parameters
#

#Called as "EvenBiggerClassifier" from Baseline model searching in notebook
class BaselineClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1: Very thin feature detector
            # Input: 1 -> Output: 4 channels
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(4), # Keeps convergence fast
            nn.ReLU(),
            nn.MaxPool1d(4),

            # Layer 2: Minimal expansion
            # Input: 4 -> Output: 8 channels
            # We use a smaller kernel (3) here to save weights
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )

        # Global Max Pooling: "Did the feature appear?"
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier: Maps just 8 numbers to your 6 classes
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)