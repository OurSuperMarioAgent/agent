import torch.nn as nn
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, is_image_space
from stable_baselines3.common.torch_layers import NatureCNN


#基础CNN实现，参考SB3的实现(PPO->ActorCriticPolicy->NatureCNN)
class MarioCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
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

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


from stable_baselines3.common import torch_layers

# 保存原始的
original_nature_cnn = torch_layers.NatureCNN

# 创建修改版
class CustomCNN(original_nature_cnn):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        # 在这里修改网络结构
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # 修改这里
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),  # 添加一层
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1)  # 添加dropout
        )

# 临时替换
torch_layers.NatureCNN = CustomCNN



class RobustCustomCNN(BaseFeaturesExtractor):
    """更健壮的自定义CNN，自动处理各种输入格式"""

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # 自动检测输入形状
        self.n_channels = self._detect_channels(observation_space)

        print(f"Detected observation space: {observation_space.shape}")
        print(f"Using {self.n_channels} input channels")

        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Flatten(),
        )

        # 计算特征维度
        with th.no_grad():
            sample_input = th.zeros(1, self.n_channels, 84, 84)  # 假设84x84输入
            n_flatten = self.cnn(sample_input).shape[1]
            print(f"CNN output features: {n_flatten}")

        self.linear = nn.Linear(n_flatten, features_dim)

    def _detect_channels(self, observation_space):
        """自动检测通道位置"""
        shape = observation_space.shape
        if len(shape) == 3:
            # 可能是 (height, width, channels) 或 (channels, height, width)
            if shape[0] in [1, 3, 4]:  # 如果第一个维度是小数字，可能是通道数
                return shape[0]
            elif shape[2] in [1, 3, 4]:  # 如果最后一个维度是小数字，可能是通道数
                return shape[2]
            else:
                # 默认假设为 (channels, height, width)
                return shape[0]
        else:
            raise ValueError(f"Unexpected observation space shape: {shape}")

    def forward(self, observations):
        # 自动处理输入格式
        if len(observations.shape) == 4:
            if observations.shape[-1] == self.n_channels:  # channels_last
                observations = observations.permute(0, 3, 1, 2)
            elif observations.shape[1] == self.n_channels:  # channels_first
                pass  # 已经是正确格式
            else:
                # 尝试自动检测
                if observations.shape[1] in [1, 3, 4]:
                    pass  # 假设已经是channels_first
                else:
                    observations = observations.permute(0, 3, 1, 2)

        return self.linear(self.cnn(observations))

