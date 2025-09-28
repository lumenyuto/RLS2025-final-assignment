import torch
from torch import nn
from utils import reparameterize

class PPOActor(nn.Module):
    def __init__(self, cnn_output_dim, action_shape):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, features):
        return torch.tanh(self.fc(features))

    def sample(self, features):
        return reparameterize(self.fc(features), self.log_stds)
