from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from super_parameters import *
from stable_baselines3.common.callbacks import EvalCallback
from custom_cnn import CustomCNN
from custom_ac import CustomACCNNPolicy
import os
import torch
from stable_baselines3.common.vec_env import VecTransposeImage

# 自定义奖励 + 栈堆叠 + 自定义CNN + 自定义激活函数（可学习参数） + 自定义策略网络 + 动态lr,clip

def lr_schedule(p):
    lr = 3e-4 * (p**2)
    return max(lr, 1e-6)

def clip_schedule(p):
    return 0.15 * p + 0.05

def train(save_path: str, total_timesteps: int = 1e6, load_model_path: str = None):
    train_env = SubprocVecEnv([make_env(rank=i, seed = 42, eval_mode=False) for i in range(n_pipe)])
    train_env = VecFrameStack(train_env, n_frame_stack)
    train_env = VecTransposeImage(train_env)
    
    eval_env = DummyVecEnv([make_env(rank=0, seed=100, eval_mode=True)])
    eval_env = VecFrameStack(eval_env, n_frame_stack)
    eval_env = VecTransposeImage(eval_env)
    
    call_back = EvalCallback(eval_env, best_model_save_path='best_models/model1/',
                             log_path='logs/', eval_freq=evaluate_frequency//n_pipe,
                             n_eval_episodes=5, deterministic=False, verbose=1)
    
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": dict(features_dim=512),
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    
    if load_model_path is not None and os.path.exists(load_model_path):
        print(f"加载已有模型：{load_model_path}，开始续训...")
        model = PPO.load(
            load_model_path,
            env=train_env,
            learning_rate=lr_schedule,
            clip_range=clip_schedule,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log="logs/",
            device=device,
        )
    else:
        print("未指定模型路径/路径不存在，新建模型从头训练...")
        model = PPO(
            CustomACCNNPolicy,
            train_env,
            verbose=1,
            tensorboard_log="logs/",
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            clip_range=clip_schedule,
            ent_coef=ent_coef,
            device=device,
        )
    model.learn(total_timesteps=total_timesteps, callback=call_back, reset_num_timesteps=False if load_model_path is not None and os.path.exists(load_model_path) else True)
    model.save(save_path)
    print(f"模型已保存至：{save_path}")
    train_env.close()
    eval_env.close()
    
if __name__ == "__main__":
    train(save_path = "final_models/final_model1.zip")