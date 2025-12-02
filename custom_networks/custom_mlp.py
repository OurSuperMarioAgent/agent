import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import MlpExtractor

#原文：(PPO->ActorCriticPolicy->MLPExtractor)

class CustomMLPExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device="auto"):
        # 基于传入的net_arch进行增强，而不是完全替换
        if isinstance(net_arch, dict):
            enhanced_net_arch = {
                'pi': net_arch.get('pi', []) + [128],  # 在原有基础上添加一层
                'vf': net_arch.get('vf', [256])  # 确保至少有一层
            }
        else:
            enhanced_net_arch = dict(pi=[256, 128], vf=[256])

        super().__init__(feature_dim, enhanced_net_arch, activation_fn, device)

        # 只在维度匹配时添加残差连接
        if feature_dim == self.latent_dim_pi:
            self.use_residual = True
        else:
            self.use_residual = False

    def forward(self, features: th.Tensor):
        shared_latent = self.shared_net(features)
        policy_latent = self.policy_net(shared_latent)
        value_latent = self.value_net(shared_latent)

        if self.use_residual:
            # 直接从shared_latent添加残差，而不是原始features
            policy_latent = policy_latent + 0.1 * shared_latent

        return policy_latent, value_latent