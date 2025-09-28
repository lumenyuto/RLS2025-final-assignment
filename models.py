import torch
from torch import nn
from utils import reparameterize # utils.pyからインポート

class CNN(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
    def forward(self, states):
        return self.net(states)

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
