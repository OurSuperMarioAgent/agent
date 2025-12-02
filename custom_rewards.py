import gym
import numpy as np
from typing import Dict, Any

#与SkipFrameWrapper处理跳帧有矛盾，暂时不要用
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, use_skip=True):
        """
        use_skip: 如果为True，表示上游已经有SkipFrameWrapper
                  奖励计算会考虑这个因素
        """
        super().__init__(env)
        self.use_skip = use_skip
        self.skip_factor = 4 if use_skip else 1  # 假设跳4帧

    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)

        # 关键：不直接使用original_reward，而是基于游戏状态计算
        x_pos = info.get('x_pos', 0)
        time_left = info.get('time', 400)

        # 重置累积奖励，基于状态重新计算
        custom_reward = 0

        # 1. 计算每帧的进展（考虑跳帧）
        if hasattr(self, 'last_x'):
            progress_per_frame = (x_pos - self.last_x) / self.skip_factor
            if progress_per_frame > 0:
                custom_reward += progress_per_frame * 0.05

        self.last_x = x_pos

        # 2. 特别事件奖励（这些不受跳帧影响）
        if info.get('flag_get', False):
            custom_reward += 10.0

        # 返回重新计算的奖励，不是累加！
        return obs, custom_reward, done, info


class SparseToDenseRewardWrapper(gym.Wrapper):
    """将稀疏奖励转换为密集奖励"""

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        # 添加生存奖励
        if not done:
            reward += 0.01  # 每步生存奖励

        # 添加时间惩罚
        reward -= 0.001  # 鼓励快速完成

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)