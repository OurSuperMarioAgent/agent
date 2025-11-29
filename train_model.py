from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import create_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from super_parameters import *
from stable_baselines3.common.callbacks import EvalCallback
import os
# 从networks模块导入所有自定义组件
from custom_networks import (
    CustomCNN,
    CustomMLPExtractor,
    CustomOptimizer,
    CosineAnnealingSchedule,
    CustomActivation,
    CustomValueHeadPolicy
)
from custom_buffers import CustomRolloutBuffer
from custom_normalize import CustomVecNormalize, SimpleVecNormalize
from custom_callbacks import CustomCallback
from custom_rewards import CustomRewardWrapper


def train(save_path: str, total_timesteps: int = 1e7):
    # 使用 DummyVecEnv 替代 SubprocVecEnv 避免多进程问题
    vec_env = DummyVecEnv([create_env for _ in range(n_pipe)])
    vec_env = VecFrameStack(vec_env, n_stack=n_frame_stacks)

    call_back = EvalCallback(vec_env, best_model_save_path='models/',
                             log_path='logs/', eval_freq=evaluate_frequency / n_pipe)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO('CnnPolicy', vec_env, verbose=1,
                tensorboard_log="logs/",
                policy_kwargs=policy_kwargs,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_episodes,
                gamma=gamma,
                learning_rate=learning_rate,
                ent_coef=ent_coef)

    model.learn(total_timesteps=total_timesteps, callback=call_back)
    model.save(save_path)