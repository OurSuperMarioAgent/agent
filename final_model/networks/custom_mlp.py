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
        # 基于传入的net_arch进行增强，修复维度匹配问题
        if isinstance(net_arch, dict):
            pi_layers = net_arch.get('pi', [64])  # 不再强制512，使用传入的或默认64
            vf_layers = net_arch.get('vf', [64])  # 不再强制512，使用传入的或默认64

            # 增强网络架构（在原有基础上扩展）
            # 策略网络：添加一层，大小为原最大层的2倍（最小128）
            if pi_layers:
                max_pi = max(pi_layers)
                enhanced_pi = pi_layers + [max(max_pi * 2, 128)]
            else:
                enhanced_pi = [64, 128]

            # 价值网络：确保至少有一层256神经元
            if vf_layers:
                enhanced_vf = vf_layers
                if not any(layer >= 256 for layer in vf_layers):
                    enhanced_vf.append(256)
            else:
                enhanced_vf = [64, 256]

            enhanced_net_arch = {
                'pi': enhanced_pi,
                'vf': enhanced_vf
            }
        else:
            # 如果不是字典格式，使用父类默认处理
            enhanced_net_arch = dict(pi=[64, 128], vf=[64, 256])

        super().__init__(feature_dim, [enhanced_net_arch], activation_fn, device)

        # 检查是否可以添加残差连接
        # 注意：这里检查的是共享层输出维度与策略网络输出维度是否匹配
        # 而不是与feature_dim匹配
        if hasattr(self, 'shared_net') and len(self.shared_net) > 0:
            # 如果有共享层，检查共享层输出维度
            shared_output_dim = self.latent_dim_pi if self.shared_net else feature_dim
            self.use_residual = (self.latent_dim_pi == shared_output_dim)
        else:
            # 如果没有共享层，检查feature_dim与策略网络输出维度
            self.use_residual = (self.latent_dim_pi == feature_dim)

    def forward(self, features: th.Tensor):
        shared_latent = self.shared_net(features)
        policy_latent = self.policy_net(shared_latent)
        value_latent = self.value_net(shared_latent)

        if self.use_residual:
            # 添加残差连接：从shared_latent到policy_latent
            policy_latent = policy_latent + 0.1 * shared_latent

        return policy_latent, value_latent