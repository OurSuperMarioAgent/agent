import torch.nn as nn
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, is_image_space

#基础CNN实现，参考SB3的实现(PPO->ActorCriticPolicy->NatureCNN)
class MarioCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations):
        return self.linear(self.cnn(observations))


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # 输入: [batch_size, 4, 84, 84]
        self.cnn = nn.Sequential(
            # 第一层：保持和原版相似
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # 第二层：适度加深
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # 第三层：增加通道但不增加复杂度
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            # 第四层：新增一层但保持简单
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        # 计算展平后的维度
        with torch.no_grad():
            n_channels = observation_space.shape[0]
            sample_input = torch.zeros(1, n_channels, *observation_space.shape[1:3])
            n_flatten = self.cnn(sample_input).shape[1]

        # 线性层：保持简单
        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations):
        # 确保通道顺序正确
        if observations.shape[-1] == 4:  # channels last
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))


class RobustCustomCNN(BaseFeaturesExtractor):
    """更健壮的自定义CNN，自动处理各种输入格式"""

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # 自动检测输入形状
        self.n_channels = self._detect_channels(observation_space)

        print(f"Detected observation space: {observation_space.shape}")
        print(f"Using {self.n_channels} input channels")

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Flatten(),
        )

        # 计算特征维度
        with torch.no_grad():
            sample_input = torch.zeros(1, self.n_channels, 84, 84)  # 假设84x84输入
            n_flatten = self.cnn(sample_input).shape[1]
            print(f"CNN output features: {n_flatten}")

        self.linear = nn.Linear(n_flatten, features_dim)

    def _detect_channels(self, observation_space):
        """自动检测通道位置"""
        shape = observation_space.shape
        if len(shape) == 3:
            # 可能是 (height, width, channels) 或 (channels, height, width)
            if shape[0] in [1, 3, 4]:  # 如果第一个维度是小数字，可能是通道数
                return shape[0]
            elif shape[2] in [1, 3, 4]:  # 如果最后一个维度是小数字，可能是通道数
                return shape[2]
            else:
                # 默认假设为 (channels, height, width)
                return shape[0]
        else:
            raise ValueError(f"Unexpected observation space shape: {shape}")

    def forward(self, observations):
        # 自动处理输入格式
        if len(observations.shape) == 4:
            if observations.shape[-1] == self.n_channels:  # channels_last
                observations = observations.permute(0, 3, 1, 2)
            elif observations.shape[1] == self.n_channels:  # channels_first
                pass  # 已经是正确格式
            else:
                # 尝试自动检测
                if observations.shape[1] in [1, 3, 4]:
                    pass  # 假设已经是channels_first
                else:
                    observations = observations.permute(0, 3, 1, 2)

        return self.linear(self.cnn(observations))

