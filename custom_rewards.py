import gym
import numpy as np
from typing import Dict, Any


class CustomRewardWrapper(gym.Wrapper):
    """
    自定义奖励函数包装器
    针对Mario游戏优化的奖励设计
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_x = 0
        self.previous_time = 400
        self.previous_score = 0
        self.previous_coins = 0
        self.previous_state = None
        self.steps_without_progress = 0
        self.max_x = 0

    def reset(self, **kwargs):
        """重置环境并初始化状态"""
        obs = self.env.reset(**kwargs)
        self.previous_x = 0
        self.previous_time = 400
        self.previous_score = 0
        self.previous_coins = 0
        self.previous_state = None
        self.steps_without_progress = 0
        self.max_x = 0
        return obs

    def step(self, action):
        """执行动作并返回自定义奖励"""
        obs, original_reward, done, info = self.env.step(action)

        # 提取游戏信息
        x_pos = info.get('x_pos', 0)
        time = info.get('time', 400)
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        status = info.get('status', 'small')

        # 初始化自定义奖励
        custom_reward = 0.0

        # 1. 前进奖励（最重要的奖励）
        progress = x_pos - self.previous_x
        if progress > 0:
            custom_reward += progress * 0.1  # 前进奖励
            self.max_x = max(self.max_x, x_pos)
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

        # 2. 停滞惩罚
        if self.steps_without_progress > 50:
            custom_reward -= 0.1  # 长期停滞惩罚

        # 3. 时间奖励/惩罚
        time_change = time - self.previous_time
        if time_change < 0:  # 时间减少（好）
            custom_reward += 0.01
        else:  # 时间增加（通常不好）
            custom_reward -= 0.005

        # 4. 分数奖励
        score_change = score - self.previous_score
        if score_change > 0:
            custom_reward += score_change * 0.001

        # 5. 金币奖励
        coins_change = coins - self.previous_coins
        if coins_change > 0:
            custom_reward += coins_change * 0.1

        # 6. 死亡惩罚
        if done and info.get('life', 2) == 0:
            custom_reward -= 15.0  # 死亡重罚

        # 7. 通关奖励
        if done and x_pos > 3000:  # 假设通关位置
            custom_reward += 50.0

        # 8. 探索奖励（基于状态新颖性）
        state_key = (x_pos // 10, status)  # 简化的状态表示
        if state_key != self.previous_state:
            custom_reward += 0.01  # 探索新状态奖励
            self.previous_state = state_key

        # 9. 速度奖励（鼓励快速移动）
        if progress > 2:  # 快速移动
            custom_reward += 0.05

        # 更新状态
        self.previous_x = x_pos
        self.previous_time = time
        self.previous_score = score
        self.previous_coins = coins

        # 组合奖励（保持原始奖励的权重）
        total_reward = original_reward + custom_reward

        # 奖励裁剪（可选）
        total_reward = np.clip(total_reward, -5.0, 5.0)

        return obs, total_reward, done, info


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