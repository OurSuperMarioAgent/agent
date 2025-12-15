import torch.nn as nn
from .custom_activation import ParamMish

class CustomValueHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(ParamMish())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # 输出状态价值
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)
