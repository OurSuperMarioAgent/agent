import time

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common import torch_layers
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from custom_networks import CustomCNN
from env_preprocess import create_env
from super_parameters import n_frame_stacks


def test(
        model_path: str,
        total_timesteps: int = 1000,
        fps: int = 20
):
    env = DummyVecEnv([create_env])
    env = VecFrameStack(env, n_stack=n_frame_stacks)

    torch_layers.NatureCNN = CustomCNN
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )
    model = PPO.load(model_path, env=env, custom_objects={
        "policy_kwargs": policy_kwargs,
    })
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
    test(model_path="models/CNN_optim_model_1.zip")