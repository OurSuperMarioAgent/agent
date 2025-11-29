import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomMLPExtractor(BaseFeaturesExtractor):
    """自定义MLP特征提取器"""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(256, features_dim)
        )

    def forward(self, observations):
        return self.net(observations)