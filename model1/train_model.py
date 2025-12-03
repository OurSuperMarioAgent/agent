from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import create_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from super_parameters import *
from stable_baselines3.common.callbacks import EvalCallback

def train(save_path: str, total_timesteps: int = 1e6):
    vec_env = SubprocVecEnv([create_env for _ in range(n_pipe)])
    
    call_back = EvalCallback(vec_env, best_model_save_path='best_models/model1/',
                             log_path='logs/', eval_freq=evaluate_frequency/n_pipe,)
    
    model = PPO('CnnPolicy', vec_env, verbose=1
                ,tensorboard_log="logs/"
                ,n_steps=n_steps
                ,batch_size=batch_size
                ,n_epochs=n_episodes
                ,gamma=gamma
                ,learning_rate=learning_rate
                ,ent_coef=ent_coef
    )
    model.learn(total_timesteps=total_timesteps, callback=call_back)
    model.save(save_path)
    
if __name__ == "__main__":
    train("final_models/final_model1.zip")