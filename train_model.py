from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common import torch_layers
from stable_baselines3.common.torch_layers import NatureCNN

from env_preprocess import create_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
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
    CustomValueHeadPolicy, CustomACCNNPolicy
)
from custom_buffers import CustomRolloutBuffer
from custom_normalize import SimpleVecNormalize
from custom_callbacks import CustomCallback
from custom_rewards import CustomRewardWrapper


def train(save_path: str, total_timesteps: int = 1e5, load_model: str = None):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 创建训练环境
    vec_env = DummyVecEnv([lambda: create_env(is_training=True, use_monitor=False)
                           for _ in range(n_pipe)])
    vec_env = VecFrameStack(vec_env, n_stack=n_frame_stacks)
    vec_env = SimpleVecNormalize(
        vec_env,
        norm_obs=False,  # 图像不应该标准化
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    vec_env = VecTransposeImage(vec_env)  # 转换为CHW格式

    # 创建评估环境（必须和训练环境完全一样的包装）
    def make_eval_env():
        return create_env(is_training=False, use_monitor=True)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=n_frame_stacks)
    eval_env = SimpleVecNormalize(
        eval_env,
        norm_obs=False,  # 和训练环境一致
        norm_reward=False,  # 评估时不标准化奖励
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    eval_env = VecTransposeImage(eval_env)  # 和训练环境一致

    def create_custom_optimizer(params, lr):
        return CustomOptimizer(
            params,
            lr=lr,
            max_grad_norm=0.8,  # 适中的梯度裁剪
            weight_decay=0.01,  # AdamW的权重衰减
            betas=(0.9, 0.999),  # Adam的动量参数
            eps=1e-8
        )

    # 改进的学习率调度（余弦退火）
    def lr_schedule(p):
        # p 是剩余进度 (从1到0)
        # 使用你的CosineAnnealingSchedule
        lr_scheduler = CosineAnnealingSchedule(
            initial_lr=3e-4,
            min_lr=1e-6
        )
        return lr_scheduler(p)

    # 改进的clip_range调度
    def clip_schedule(p):
        # 初期保守，中期适中，后期严格
        if p > 0.7:  # 前30%训练
            return 0.15
        elif p > 0.3:  # 中间40%
            return 0.12
        else:  # 最后30%
            return 0.08

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

        model.policy.optimizer = create_custom_optimizer(
            model.policy.parameters(),
            lr=model.policy.optimizer.param_groups[0]['lr']
        )

    else:
        print("创建新模型")

        torch_layers.NatureCNN = CustomCNN
        policy_kwargs = dict(
            features_extractor_class=NatureCNN,
            features_extractor_kwargs=dict(features_dim=512),
            optimizer_class=CustomOptimizer,
            optimizer_kwargs={
                "max_grad_norm": 0.8,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8
            }
        )

        model = PPO(CustomACCNNPolicy, vec_env, verbose=1,
                    tensorboard_log="logs/",
                    policy_kwargs=policy_kwargs,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_episodes,
                    gamma=gamma,
                    learning_rate=lr_schedule,
                    ent_coef=ent_coef,
                    clip_range=clip_schedule,
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