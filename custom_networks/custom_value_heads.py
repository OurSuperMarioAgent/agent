import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticCnnPolicy


class CustomValueHead(nn.Module):
    """自定义价值函数头"""

    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),

            nn.Linear(hidden_dim // 2, 1)  # 输出状态价值
        )

    def forward(self, features):
        return self.net(features)


class CustomValueHeadPolicy(ActorCriticCnnPolicy):
    """使用自定义价值函数头的策略"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换价值网络
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        self.value_net = CustomValueHead(latent_dim_vf)