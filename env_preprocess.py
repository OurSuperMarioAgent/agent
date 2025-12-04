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
    if use_monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)

    return env