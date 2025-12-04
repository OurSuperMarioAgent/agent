from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common import torch_layers
from stable_baselines3.common.torch_layers import NatureCNN

from env_preprocess import create_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from super_parameters import *
from stable_baselines3.common.callbacks import EvalCallback
import os
import shutil
# 从networks模块导入所有自定义组件
from custom_networks import (
    MarioCNN,
    CustomCNN,
    CustomMLPExtractor,
    CustomOptimizer,
    CosineAnnealingSchedule,
    SimpleResBlock,
    CustomValueHeadPolicy
)
from custom_buffers import CustomRolloutBuffer
from custom_normalize import CustomVecNormalize, SimpleVecNormalize
from custom_callbacks import CustomCallback
from custom_rewards import CustomRewardWrapper

def train(save_path: str, total_timesteps: int = 5e5):
    log_dir = "logs"

    # if os.path.exists(log_dir):
    #     # 删除整个目录
    #     shutil.rmtree(log_dir)
    #
    # os.makedirs(log_dir, exist_ok=True)

    # 使用 DummyVecEnv 替代 SubprocVecEnv 避免多进程问题
    vec_env = DummyVecEnv([lambda: create_env(is_training=True, use_monitor=False)
                           for _ in range(n_pipe)])
    vec_env = VecFrameStack(vec_env, n_stack=n_frame_stacks)

    def make_eval_env():
        # 评估环境需要Monitor来正确记录奖励
        return create_env(is_training=False, use_monitor=True)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=n_frame_stacks)

    call_back = EvalCallback(
        eval_env,
        best_model_save_path='models/',
        log_path='logs/',
        eval_freq=max(evaluate_frequency // n_pipe, 1),  # 确保整数
        n_eval_episodes=5,  # 评估5个episode取平均
        deterministic=False,  # 评估时也带一点随机性
        render=False,
        verbose=1  # 显示评估信息
    )

    #torch_layers.NatureCNN = CustomCNN
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
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