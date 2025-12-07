from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import create_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from super_parameters import *
from stable_baselines3.common.callbacks import EvalCallback
from custom_cnn import CustomCNN
from custom_ac import CustomACCNNPolicy

# 自定义奖励 + 栈堆叠 + 自定义CNN + 自定义激活函数（可学习参数） + 自定义策略网络
def train(save_path: str, total_timesteps: int = 1e6):
    vec_env = SubprocVecEnv([create_env for _ in range(n_pipe)])
    vec_env = VecFrameStack(vec_env, n_frame_stack)
    
    call_back = EvalCallback(vec_env, best_model_save_path='best_models/model1/',
                             log_path='logs/', eval_freq=evaluate_frequency/n_pipe,)
    
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": dict(features_dim=512),
    }
    
    model = PPO(CustomACCNNPolicy, vec_env, verbose=1
                ,tensorboard_log="logs/"
                ,n_steps=n_steps
                ,batch_size=batch_size
                ,n_epochs=n_epochs
                ,gamma=gamma
                ,learning_rate=learning_rate
                ,ent_coef=ent_coef
                ,policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=total_timesteps, callback=call_back)
    model.save(save_path)
    
if __name__ == "__main__":
    train("final_models/final_model1.zip")