import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import MlpExtractor

#原文：(PPO->ActorCriticPolicy->MLPExtractor)(自定义的多层感知机特征提取器)

# 基础功能：从CNN特征提取器的输出（feature_dim维向量）中提取策略和价值特征
# 增强功能：在原有网络架构基础上自动扩展层数并添加残差连接
# 主要目的：增强网络表达能力，防止梯度消失，提高学习稳定性

# 输入示例：net_arch = {'pi': [64], 'vf': [64]}
# 输出结果：enhanced_net_arch = {'pi': [64, 128], 'vf': [256]}
# 策略网络 (pi)：在原架构末尾加一层128神经元
# 价值网络 (vf)：确保至少有一层256神经元

class CustomMLPExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device="auto"):
        # 基于传入的net_arch进行增强，而不是完全替换
        # 修复：确保网络架构与feature_dim匹配
        if isinstance(net_arch, dict):
            pi_layers = net_arch.get('pi', [512, 256])  # 第一层必须是512！
            vf_layers = net_arch.get('vf', [512, 256])  # 第一层必须是512！

            enhanced_net_arch = {
                'pi': pi_layers,
                'vf': vf_layers
            }
        else:
            enhanced_net_arch = dict(pi=[512, 256], vf=[512, 256])  # 都从512开始

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