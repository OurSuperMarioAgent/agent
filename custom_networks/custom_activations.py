import torch.nn as nn
import torch


class CustomActivation(nn.Module):
    """自定义激活函数 - Swish变体"""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)