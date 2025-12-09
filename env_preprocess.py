import gym
import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,ResizeObservation

from custom_crop_wrapper import CropObservation
from custom_rewards import CustomRewardWrapper
from skip_frame_wrapper import SkipFrameWrapper
from stable_baselines3 import PPO
from super_parameters import *


def create_env(is_training=True, use_monitor=False):
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = CropObservation(env, crop_box)
    env = ResizeObservation(env, resize_observation_shape)
    env = SkipFrameWrapper(env, skip=n_frame_skip)
    env = CustomRewardWrapper(env, use_skip=True)

    if use_monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)

    return env

def make_env(rank: int, seed: int = 42, eval_mode: bool = False):
    """
    精简版子环境创建：仅传递训练seed，eval模式无需特殊处理
    """
    def _init():
        # 训练模式：每个子环境用不同seed（seed + rank），保证多样性
        env_seed = seed + rank if not eval_mode else None
        env = create_env(seed=env_seed, eval_mode=eval_mode)
        return env
    return _init