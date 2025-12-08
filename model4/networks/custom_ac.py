from stable_baselines3.common.policies import ActorCriticCnnPolicy
from .custom_policy_net import CustomPolicyHead
from .custom_value_net import CustomValueHead
from functools import partial
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule

class CustomACCNNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.action_dist, CategoricalDistribution)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        
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
            self.action_net.apply(partial(self.init_weights, gain=0.01))
            self.value_net.apply(partial(self.init_weights, gain=1.0))
        
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(0),
            **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(action_logits=self.action_net(latent_pi))

    def _get_values_from_latent(self, latent_vf):
        return self.value_net(latent_vf)