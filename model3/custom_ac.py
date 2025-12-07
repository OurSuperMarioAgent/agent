from stable_baselines3.common.policies import ActorCriticCnnPolicy
from custom_policy_net import CustomPolicyHead
from custom_value_net import CustomValueHead
from functools import partial

class CustomACCNNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        latent_dim_vf = self.mlp_extractor.latent_dim_vf

        # 替换默认策略头和价值头
        self.action_net = CustomPolicyHead(latent_dim_pi, hidden_dims=[512, 256], action_dim=self.action_space.n)
        self.value_net = CustomValueHead(latent_dim_vf, hidden_dims=[512, 256])

        # 初始化
        self.action_net.apply(partial(self.init_weights, gain=0.01))
        self.value_net.apply(partial(self.init_weights, gain=1.0))