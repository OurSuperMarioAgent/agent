import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,ResizeObservation
from skip_frame_wrapper import SkipFrameWrapper
from stable_baselines3 import PPO
from super_parameters import *
from reward_wrapper import CustomRewardWrapper, SparseToDenseRewardWrapper

def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, resize_observation_shape)
    
    # 在base model的训练中加入跳帧，只是加速收敛，不影响最终效果
    env = SkipFrameWrapper(env, skip=n_frame_skip)
    
    # 创新点：自定义奖励函数
    # 1. 将稀疏奖励转换为密集奖励
    env = SparseToDenseRewardWrapper(env)
    # 2. 给mario更多更明确的引导
    env = CustomRewardWrapper(env)
    return env