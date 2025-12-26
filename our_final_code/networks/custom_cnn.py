import torch.nn as nn
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, is_image_space
from stable_baselines3.common.torch_layers import NatureCNN
from .custom_res_block import SimpleResBlock
from .custom_activation import ParamMish

class MarioCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


from stable_baselines3.common import torch_layers

original_nature_cnn = torch_layers.NatureCNN

class CustomCNN(original_nature_cnn):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.GroupNorm(4, 32),
            ParamMish(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.GroupNorm(8, 64),
            ParamMish(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.GroupNorm(8, 64),
            ParamMish(),
            
            SimpleResBlock(64),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            ParamMish(),

            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.LayerNorm(features_dim),
            ParamMish(),
        )
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))