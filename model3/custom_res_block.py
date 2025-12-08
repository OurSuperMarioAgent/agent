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

        # 通道注意力（放在最后）
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 16), 1),  # 确保至少4个通道
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // 16), channels, 1),
            nn.Sigmoid()
        )

        # 初始化最后层接近0
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        identity = x

        # 第一层：预激活
        out = self.norm1(x)
        out = nn.ReLU(inplace=True)(out)
        out = self.conv1(out)

        # 第二层：预激活
        out = self.norm2(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.conv2(out)

        # 通道注意力
        attention = self.attention(out)
        out = out * attention

        # 最后再加一个ReLU（可选）
        out = nn.ReLU(inplace=True)(out)

        # 残差连接（权重可以大一些）
        return identity + 0.5 * out