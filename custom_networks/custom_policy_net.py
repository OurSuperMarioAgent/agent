import torch.nn as nn

class CustomPolicyHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], action_dim=7):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)