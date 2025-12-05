import gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,ResizeObservation

from custom_rewards import CustomRewardWrapper
from skip_frame_wrapper import SkipFrameWrapper
from stable_baselines3 import PPO
from super_parameters import *

def create_env(is_training=True, use_monitor=False):
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, resize_observation_shape)
    env = SkipFrameWrapper(env, skip=n_frame_skip)
    env = CustomRewardWrapper(env, use_skip=True)

    def random_start_wrapper(env, start_positions=[0, 50, 100, 200]):
        class RandomStartEnv(gym.Wrapper):
            def reset(self, **kwargs):
                obs = super().reset(**kwargs)
                # 随机跳过一些初始步骤
                if np.random.random() < 0.3:  # 30%概率从不同位置开始
                    for _ in range(np.random.choice(start_positions)):
                        obs, _, _, _ = self.step(0)  # 执行无操作
                return obs

        return RandomStartEnv(env)
    #env = random_start_wrapper(env)

    if use_monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)

    return env