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
        self.ent_coef_schedule = kwargs.pop("entropy_coef_schedule", None)
        
        if net_arch is None:
            self.custom_net_arch = dict(pi=[64, 128], vf=[64, 256])
        elif isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            self.custom_net_arch = net_arch[0]
        elif isinstance(net_arch, dict):
            self.custom_net_arch = net_arch
        else:
            self.custom_net_arch = dict(pi=net_arch, vf=net_arch)
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        
        self.action_net = CustomPolicyHead(
            input_dim=latent_dim_pi,
            hidden_dims=[512, 256],
            action_dim=self.action_space.n
        ).to(self.device)
        
        self.value_net = CustomValueHead(
            input_dim=latent_dim_vf,
            hidden_dims=[512, 256]
        ).to(self.device)
        
        if self.ortho_init:
            self.mlp_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.action_net.apply(partial(self.init_weights, gain=0.1))
            self.value_net.apply(partial(self.init_weights, gain=1.2))
        
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
        """使用自定义的策略头"""
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)
    
    def _get_values_from_latent(self, latent_vf):
        """使用自定义的价值头"""
        return self.value_net(latent_vf)