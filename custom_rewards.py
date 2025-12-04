import gym
import numpy as np
from typing import Dict, Any, Tuple


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, use_skip=True):
        super().__init__(env)
        self.use_skip = use_skip
        self.skip_factor = 4 if use_skip else 1

        # 状态追踪
        self.last_x = 0
        self.max_x = 0
        self.episode_reward = 0

        # 关键：使用合理的奖励值，不进行过度处理
        self.config = {
            'progress_multiplier': 0.5,  # 每像素0.5奖励
            'coin_reward': 8.0,  # 每个金币10分
            'flag_reward': 300.0,  # 过关300分
            'kill_reward': 5.0,  # 杀敌5分
            'time_penalty': 0.02,  # 时间惩罚
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, original_reward, done, info = self.env.step(action)

        x_pos = info.get('x_pos', 0)
        total_reward = original_reward  # 保留原始奖励

        # 1. 进展奖励
        if hasattr(self, 'last_x'):
            progress = x_pos - self.last_x
            if progress > 0:
                total_reward += progress * self.config['progress_multiplier']

                # 突破最远距离额外奖励
                if x_pos > self.max_x:
                    bonus = (x_pos - self.max_x) * 0.1
                    total_reward += bonus
                    self.max_x = x_pos

        # 2. 金币奖励
        coins = info.get('coins', 0)
        if coins > getattr(self, 'last_coins', 0):
            coins_gained = coins - getattr(self, 'last_coins', 0)
            total_reward += coins_gained * self.config['coin_reward']

        # 3. 过关奖励（比原来500小，但更合理）
        if info.get('flag_get', False):
            total_reward += self.config['flag_reward']

        # 4. 杀敌奖励
        if info.get('enemy_killed', False):
            total_reward += self.config['kill_reward']

        # 5. 时间惩罚
        total_reward -= self.config['time_penalty']

        # 6. 死亡惩罚
        if done and info.get('life', 3) < getattr(self, 'last_life', 3):
            total_reward -= 50.0

        # 更新状态
        self.last_x = x_pos
        self.last_coins = coins
        self.last_life = info.get('life', 3)
        self.episode_reward += total_reward

        # 简单调试
        if done:
            print(f"回合结束: X={x_pos}, 总奖励={self.episode_reward:.1f}")

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_x = 0
        self.max_x = 0
        self.episode_reward = 0
        print("新回合开始")
        return obs