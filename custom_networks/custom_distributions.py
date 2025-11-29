import torch
from stable_baselines3.common.distributions import CategoricalDistribution


class CustomDistribution(CategoricalDistribution):
    """自定义动作分布，添加熵奖励调整"""

    def __init__(self, action_dim):
        super().__init__(action_dim)

    def entropy(self):
        """重写熵计算，可添加自定义逻辑"""
        base_entropy = super().entropy()
        # 例如：对熵进行缩放
        return base_entropy * 0.9

    def log_prob(self, actions):
        """重写log概率计算"""
        return super().log_prob(actions)