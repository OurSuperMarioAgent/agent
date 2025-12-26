import torch.nn as nn
import torch

class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = max(1, channels // 8)

        # 预激活结构
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        # 通道注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // 16), channels, 1),
            nn.Sigmoid()
        )

        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        identity = x

        out = self.norm1(x)
        out = nn.ReLU(inplace=True)(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.conv2(out)

        # 通道注意力
        attention = self.attention(out)
        out = out * attention

        out = nn.ReLU(inplace=True)(out)

        # 残差连接
        return identity + 0.5 * out