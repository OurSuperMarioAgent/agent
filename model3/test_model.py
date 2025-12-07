import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import create_env
from stable_baselines3.common.vec_env import DummyVecEnv
from super_parameters import *

def test(
    model_path: str, 
    total_timesteps: int = 1000,
    fps: int = 20
):
    env = DummyVecEnv([create_env])

    model = PPO.load(model_path, env=env)
    delay = 2.0 / fps

    obs = env.reset()
    for i in range(int(total_timesteps)):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(delay)
        
        if any(done):
            print(f"Episode done! Info: {info}")
            obs = env.reset()
            time.sleep(1.0)

    env.close()
    
if __name__ == "__main__":
    test("best_models/model1/best_model.zip")