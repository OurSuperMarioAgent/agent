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


def train(save_path: str, total_timesteps: int = 1e5, load_model: str = None):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 创建环境
    vec_env = DummyVecEnv([lambda: create_env(is_training=True, use_monitor=False)
                           for _ in range(n_pipe)])
    vec_env = VecFrameStack(vec_env, n_stack=n_frame_stacks)

    def make_eval_env():
        return create_env(is_training=False, use_monitor=True)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=n_frame_stacks)

    call_back = EvalCallback(
        eval_env,
        best_model_save_path='models/',
        log_path='logs/',
        eval_freq=max(evaluate_frequency // n_pipe, 1),
        n_eval_episodes=5,
        deterministic=False,
        render=False,
        verbose=1
    )

    # 模型加载/创建
    if load_model and os.path.exists(load_model):
        print(f"加载模型: {load_model}")
        model = PPO.load(load_model)
        model.set_env(vec_env)
        reset_timesteps = False
        print(f"已训练步数: {model.num_timesteps}")
    else:
        print("创建新模型")
        torch_layers.NatureCNN = CustomCNN
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
                    ent_coef=ent_coef
                    )
        reset_timesteps = True

    # 训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=call_back,
        reset_num_timesteps=reset_timesteps
    )
    model.save(save_path)

    vec_env.close()
    eval_env.close()

    print(f"训练完成，模型保存到: {save_path}")