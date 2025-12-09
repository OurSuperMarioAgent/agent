import torch
import torch.nn as nn
import torch.nn.functional as F


class ParamMish(nn.Module):
    """
    可学习参数的 Mish 激活函数:
    f(x) = x * tanh(alpha * softplus(beta * x))
    """

    def __init__(self, alpha_init=1.0, beta_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return x * torch.tanh(self.alpha * F.softplus(self.beta * x))