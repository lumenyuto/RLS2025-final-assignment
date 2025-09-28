import torch
from torch import nn

class PPOCritic(nn.Module):
    def __init__(self, cnn_output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features):
        return self.fc(features)
