import gym
import numpy as np
from typing import Dict, Any, Tuple

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, use_skip=True, use_curriculum=True):
        super().__init__(env)
        self.use_skip = use_skip
        self.skip_factor = 4 if use_skip else 1
        self.use_curriculum = use_curriculum

        # 状态追踪
        self.last_x = 0
        self.max_x = 0
        self.episode_reward = 0

        # 课程学习阶段
        if use_curriculum:
            self.stages = [500, 1000, 1500, 2000, 2500, 3000]
            self.current_stage = 0
            self.stage_completed = [False] * len(self.stages)

        # 奖励参数
        self.config = {
            'progress_multiplier': 0.5,
            'coin_reward': 5.0,
            'flag_reward': 2000.0,
            'kill_reward': 3.0,
            'time_penalty': 0.03,
            'stage_base_reward': 150.0,
            'stage_progress_bonus': 50.0,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, original_reward, done, info = self.env.step(action)

        x_pos = info.get('x_pos', 0)
        total_reward = original_reward

        # 1. 进展奖励
        if hasattr(self, 'last_x'):
            progress = x_pos - self.last_x
            if progress > 0:
                total_reward += progress * self.config['progress_multiplier']
                if x_pos > self.max_x:
                    old_max = self.max_x
                    self.max_x = x_pos
                    breakthrough = x_pos - old_max
                    if breakthrough > 20:
                        bonus = breakthrough * 0.05
                        total_reward += bonus

        # 2. 课程学习阶段奖励
        if self.use_curriculum:
            for i, stage_target in enumerate(self.stages):
                if not self.stage_completed[i] and x_pos >= stage_target:
                    stage_reward = self.config['stage_base_reward'] * (i + 1)
                    total_reward += stage_reward
                    self.stage_completed[i] = True
                    self.current_stage = i + 1

                    if stage_target >= 2000:  # 只显示重要阶段
                        print(f"阶段{i + 1}: {stage_target}像素 +{stage_reward:.0f}分")

        # 3. 接近终点奖励
        if x_pos > 2500:
            proximity_bonus = (x_pos - 2500) * 0.1
            total_reward += proximity_bonus
            if x_pos > 2800 and self.last_x <= 2800:
                total_reward += 100.0

        # 4. 金币奖励
        coins = info.get('coins', 0)
        if coins > getattr(self, 'last_coins', 0):
            coins_gained = coins - getattr(self, 'last_coins', 0)
            total_reward += coins_gained * self.config['coin_reward']

        # 5. 过关奖励
        if info.get('flag_get', False):
            total_reward += self.config['flag_reward']
            print(f"通关 +{self.config['flag_reward']}分")

        # 6. 杀敌奖励
        if info.get('enemy_killed', False):
            total_reward += self.config['kill_reward']

        # 7. 时间惩罚
        total_reward -= self.config['time_penalty']

        # 8. 死亡惩罚
        if done and info.get('life', 3) < getattr(self, 'last_life', 3):
            death_penalty = 50.0
            if self.max_x > 1500:
                progress_factor = min(0.7, self.max_x / 3200)
                death_penalty *= (1 - progress_factor)
            total_reward -= death_penalty

        # 9. 更新状态
        self.last_x = x_pos
        self.last_coins = coins
        self.last_life = info.get('life', 3)
        self.episode_reward += total_reward

        # 10. 调试信息（只在重要事件时输出）
        if done:
            if info.get('flag_get', False):
                print(f"通关 X={x_pos} 奖励={self.episode_reward:.1f}")
            elif x_pos > 2500:
                print(f"失败 X={x_pos} 最远={self.max_x}")

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_x = 0
        self.max_x = 0
        self.episode_reward = 0

        if self.use_curriculum:
            self.current_stage = 0
            self.stage_completed = [False] * len(self.stages)

        return obs