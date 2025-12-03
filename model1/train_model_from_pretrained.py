from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import create_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from super_parameters import *
from stable_baselines3.common.callbacks import EvalCallback

def train_from_pretrained(save_path: str, total_timesteps: int = 8e3 + 1e4):
    vec_env = SubprocVecEnv([create_env for _ in range(n_pipe)])
    
    call_back = EvalCallback(vec_env, best_model_save_path='best_models/model1/',
                             log_path='logs/', eval_freq=evaluate_frequency/n_pipe,)
    
    model = PPO.load(save_path, env=vec_env)
    model.learn(total_timesteps=total_timesteps, callback=call_back, reset_num_timesteps=False)
    model.save(save_path)
    
if __name__ == "__main__":
    train_from_pretrained("best_models/model1/best_model.zip")