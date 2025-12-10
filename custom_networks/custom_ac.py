from stable_baselines3.common.policies import ActorCriticCnnPolicy
from torch import nn

from . import CustomMLPExtractor
from .custom_policy_net import CustomPolicyHead
from .custom_value_net import CustomValueHead
from functools import partial
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule


class CustomACCNNPolicy(ActorCriticCnnPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.ReLU,
            *args,
            **kwargs
    ):
        # 保存自定义的net_arch
        self.custom_net_arch = net_arch

        # 传递给父类时，设置使用自定义提取器
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,  # 正常传递
            activation_fn=activation_fn,
            *args,
            **kwargs
        )

        # 关键：替换父类的mlp_extractor为您的CustomMLPExtractor
        # 需要在特征提取器构建后进行
        self.mlp_extractor_class = CustomMLPExtractor

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        # 重新构建mlp_extractor，使用您的自定义类
        if hasattr(self, 'features_extractor'):
            # 获取特征维度
            feature_dim = self.features_extractor.features_dim

            # 创建自定义MLP提取器
            self.mlp_extractor = CustomMLPExtractor(
                feature_dim=feature_dim,
                net_arch=self.custom_net_arch if self.custom_net_arch else dict(pi=[64, 128], vf=[64, 256]),
                activation_fn=self.activation_fn,
                device=self.device
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
            self.action_net.apply(partial(self.init_weights, gain=0.1))
            self.value_net.apply(partial(self.init_weights, gain=1.2))

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(0),
            **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(action_logits=self.action_net(latent_pi))

    def _get_values_from_latent(self, latent_vf):
        return self.value_net(latent_vf)