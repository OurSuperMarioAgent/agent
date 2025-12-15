from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from env_preprocess import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from super_parameters import *
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from networks import CustomCNN
from networks import CustomACCNNPolicy
import torch
from stable_baselines3.common.vec_env import VecTransposeImage
from schedulers import (lr_schedule, clip_schedule, ent_schedule)
from call_backs import EntropyScheduleCallback
from algorithms import SegmentTrainingPPO

def train(save_path: str, total_timesteps: int = 5_000_000, segment_timesteps: int = 1_000_000):
    train_env = SubprocVecEnv([make_env(rank=i, seed = 42, eval_mode=False) for i in range(n_pipe)])
    train_env = VecFrameStack(train_env, n_frame_stack)
    train_env = VecTransposeImage(train_env)
    
    eval_env = DummyVecEnv([make_env(rank=0, seed=100, eval_mode=True)])
    eval_env = VecFrameStack(eval_env, n_frame_stack)
    eval_env = VecTransposeImage(eval_env)
    
    eval_call_back = EvalCallback(eval_env, best_model_save_path='results_and_models/best_models/',
                             log_path='results_and_models/logs/', eval_freq=evaluate_frequency//n_pipe,
                             n_eval_episodes=5, deterministic=True, verbose=1)
    ent_call_back = EntropyScheduleCallback(verbose=1)
    call_backs = CallbackList([eval_call_back, ent_call_back])

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": dict(features_dim=512),
        "entropy_coef_schedule": ent_schedule,
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    model = SegmentTrainingPPO(
            CustomACCNNPolicy,
            train_env,
            verbose=1,
            tensorboard_log="results_and_models/logs/",
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            learning_rate=lr_schedule,
            clip_range=clip_schedule,
            # ent_coef的初始化值不影响后续ent_coef的调度
            ent_coef=ent_coef,
            device=device,
    )
    model.learn(total_timesteps=total_timesteps, segment_timesteps=segment_timesteps, callback=call_backs)
    model.save(save_path)
    train_env.close()
    eval_env.close()
    print(f"Training complete. model saved to {save_path}")

if __name__ == "__main__":
    train("results_and_models/final_models/final_model.zip")