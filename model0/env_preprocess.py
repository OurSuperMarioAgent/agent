import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,ResizeObservation
from skip_frame_wrapper import SkipFrameWrapper
from stable_baselines3 import PPO
from super_parameters import *

def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, resize_observation_shape)
    #env = SkipFrameWrapper(env, skip=n_frame_skip)
    return env