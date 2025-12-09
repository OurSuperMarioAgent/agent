import torch
from stable_baselines3.common.distributions import CategoricalDistribution

#(带自适应熵系数的分类分布) 直接影响：不再需要手动调整ent_coef
# 原理：在PPO中，ent_coef是固定的，而在实际应用中，我们往往希望探索-利用平衡，因此需要自适应调整熵系数。
# 自适应调整熵的具体做法：在训练过程中，根据性能指标（比如回报）动态调整熵系数。

# 基础功能：标准的分类动作分布（Categorical Distribution）
# 增强功能：自适应调整熵（探索性），替代PPO中固定的ent_coef
# 主要目的：更精细地控制探索-利用平衡，避免手动调整熵系数

class CustomDistribution(CategoricalDistribution):
    def __init__(self, action_dim, initial_beta=0.01, decay_rate=0.9995):
        super().__init__(action_dim)
        self.beta = initial_beta  # 熵系数
        self.decay_rate = decay_rate
        self.train_step = 0

    def entropy(self):
        """带自适应系数的熵计算"""
        base_entropy = super().entropy()
        # 随着训练衰减熵系数
        self.beta *= self.decay_rate
        self.train_step += 1
        return base_entropy * self.beta

    def update_beta(self, performance_metric):
        """根据性能动态调整熵系数"""
        # 如果性能提升慢，增加探索
        # 动态调整：理想情况是训练初期熵高（多探索），后期熵低（多利用）
        if performance_metric < 0.1:  # 性能差
            self.beta = min(self.beta * 1.1, 0.1)  # 增加探索
        else:  # 性能好
            self.beta = max(self.beta * 0.99, 0.001)  # 减少探索

