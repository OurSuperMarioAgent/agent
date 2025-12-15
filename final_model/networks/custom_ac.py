import numpy as np
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from .custom_policy_net import CustomPolicyHead
from .custom_value_net import CustomValueHead
from functools import partial
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule
from .custom_mlp import CustomMLPExtractor
from .custom_activation import ParamMish

class CustomACCNNPolicy(ActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        net_arch=None,
        activation_fn=ParamMish,
        *args,
        **kwargs
    ):
        # 熵系数调度器
        self.ent_coef_schedule = kwargs.pop("entropy_coef_schedule", None)
        
        # ========== 新增：处理net_arch格式，适配CustomMLPExtractor ==========
        # 1. 兜底：net_arch为None时，设置默认字典
        if net_arch is None:
            self.custom_net_arch = dict(pi=[64, 128], vf=[64, 256])
        # 2. 父类默认格式（列表套字典）：提取第一个字典
        elif isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            self.custom_net_arch = net_arch[0]
        # 3. 直接传字典：直接使用
        elif isinstance(net_arch, dict):
            self.custom_net_arch = net_arch
        # 4. 其他格式（如[64,64]）：转为字典（pi/vf共用）
        else:
            self.custom_net_arch = dict(pi=net_arch, vf=net_arch)
        
        # 调用父类初始化，使用自定义的mlp_extractor
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )
        
        # 此时父类已经使用我们的CustomMLPExtractor构建了mlp_extractor
        # 现在我们可以替换策略头和价值头
        
        # 获取mlp_extractor的输出维度
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        
        # 替换策略头
        self.action_net = CustomPolicyHead(
            input_dim=latent_dim_pi,
            hidden_dims=[512, 256],
            action_dim=self.action_space.n
        ).to(self.device)
        
        # 替换价值头
        self.value_net = CustomValueHead(
            input_dim=latent_dim_vf,
            hidden_dims=[512, 256]
        ).to(self.device)
        
        # 重新初始化权重
        if self.ortho_init:
            self.mlp_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.action_net.apply(partial(self.init_weights, gain=0.1))
            self.value_net.apply(partial(self.init_weights, gain=1.2))
        
        # 重新创建优化器（因为网络参数改变了）
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1.0),
            **(self.optimizer_kwargs or {})
        )
    
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomMLPExtractor(
            self.features_dim,
            net_arch=self.custom_net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
    
    def _get_action_dist_from_latent(self, latent_pi):
        """重写这个方法以使用自定义的策略头"""
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)
    
    def _get_values_from_latent(self, latent_vf):
        """重写这个方法以使用自定义的价值头"""
        return self.value_net(latent_vf)