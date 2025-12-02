from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import create_env

def test(model_path: str, total_timesteps: int = 1e4):
    env = create_env()

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    for i in range(total_timesteps):
        obs = obs.copy()
        action,_state = model.predict(obs,deterministic=False)
        obs,reward,done,info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()