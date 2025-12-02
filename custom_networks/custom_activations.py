import torch.nn as nn
import torch


class CustomActivation(nn.Module):
    """自定义激活函数 - Swish变体"""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class MarioActivation(nn.Module):
    """针对Mario游戏优化的激活函数"""

    def __init__(self, features_dim=512):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(features_dim))  # 每个特征独立的β
        self.residual = nn.Parameter(torch.tensor(0.1))  # 残差连接权重，防止梯度消失

    def forward(self, x):
        # 每个特征有独立的非线性程度
        swish = x * torch.sigmoid(self.beta * x)
        # 残差连接保持梯度流动
        return swish + self.residual * x