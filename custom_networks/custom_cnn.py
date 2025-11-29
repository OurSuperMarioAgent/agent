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
    """修复版自定义CNN - 处理通道顺序问题"""

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # 输入形状应该是 [batch_size, 4, 84, 84] (channels_first)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
        )

        # 修复：正确计算特征维度
        with torch.no_grad():
            # 确保输入形状正确 [batch_size, channels, height, width]
            n_channels = observation_space.shape[0]  # 应该是4（帧堆叠）
            sample_input = torch.zeros(1, n_channels, *observation_space.shape[1:3])
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

    def forward(self, observations):
        # 确保输入形状正确
        # observations 应该是 [batch_size, channels, height, width]
        # 如果不是，需要转置
        if observations.shape[-1] == 4:  # 如果通道在最后
            # 从 [batch_size, height, width, channels] 转置为 [batch_size, channels, height, width]
            observations = observations.permute(0, 3, 1, 2)

        features = self.cnn(observations)
        return self.linear(features)


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


class SimpleCompatibleCNN(BaseFeaturesExtractor):
    """简单兼容的CNN，直接使用NatureCNN结构"""

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # 使用与NatureCNN相同的结构确保兼容性
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算特征维度（使用标准84x84输入）
        with torch.no_grad():
            sample_input = torch.zeros(1, 4, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # 处理通道顺序
        if observations.shape[-1] == 4:  # channels_last
            observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))