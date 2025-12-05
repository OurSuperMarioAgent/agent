from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import create_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from super_parameters import *
from stable_baselines3.common.callbacks import EvalCallback
from custom_cnn import CustomCNN

def train_from_pretrained(
    save_path: str
    ,pretrained_model_path: str
    ,total_timesteps: int = 2e6
):
    vec_env = SubprocVecEnv([create_env for _ in range(n_pipe)])
    vec_env = VecFrameStack(vec_env, n_frame_stack)
    
    call_back = EvalCallback(vec_env, best_model_save_path='best_models/model2/',
                             log_path='logs/', eval_freq=evaluate_frequency/n_pipe,)
    
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": dict(features_dim=512),
    }
    model = PPO.load(pretrained_model_path, env=vec_env, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=call_back, reset_num_timesteps=False)
    model.save(save_path)
    
if __name__ == "__main__":
    train_from_pretrained(pretrained_model_path="best_models/model1/best_model.zip", save_path="final_models/final_model2.zip")