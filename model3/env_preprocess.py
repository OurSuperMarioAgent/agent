import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,ResizeObservation
from skip_frame_wrapper import SkipFrameWrapper
from stable_baselines3 import PPO
from super_parameters import *
from reward_wrapper import CustomRewardWrapper, SparseToDenseRewardWrapper
import numpy as np
from stable_baselines3.common.monitor import Monitor
import gym
from crop_wrapper import CropObservation

def create_env(seed: int = None, eval_mode: bool = False):
    """
    精简版环境创建：保留核心预处理，移除冗余的seed手动控制
    :param seed: 训练模式下的固定seed（保证可复现）
    :param eval_mode: 评估模式下不使用自定义奖励包装器
    """
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = CropObservation(env, crop_box)
    env = ResizeObservation(env, resize_observation_shape)
    env = SkipFrameWrapper(env, n_frame_skip)
    
    if not eval_mode:
        env = SparseToDenseRewardWrapper(env)
        env = CustomRewardWrapper(env)
        if seed is not None:
            env.seed(seed)
            env.action_space.seed(seed)
    
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