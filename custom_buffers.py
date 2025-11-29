import torch
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer
from typing import Optional


class CustomRolloutBuffer(RolloutBuffer):
    """
    自定义经验回放缓冲区
    添加优势标准化和优先级采样
    """

    def __init__(self, *args, priority_alpha=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority_alpha = priority_alpha
        self.priorities = None
        self.advantages = None

    def reset(self) -> None:
        super().reset()
        self.priorities = None
        self.advantages = None

    def add(self, *args, **kwargs) -> None:
        super().add(*args, **kwargs)
        # 可以在这里添加自定义逻辑

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """自定义优势计算，添加标准化"""
        super().compute_returns_and_advantage(last_values, dones)

        # 优势标准化
        if self.advantages is not None:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # 初始化优先级（基于优势的绝对值）
        if self.priorities is None:
            self.priorities = np.ones_like(self.advantages)
        else:
            # 更新优先级，基于优势的重要性
            new_priorities = np.abs(self.advantages) + 1e-6
            self.priorities = np.maximum(self.priorities, new_priorities)

    def sample_batch_indices(self, batch_size: int) -> np.ndarray:
        """优先级采样"""
        if self.priorities is not None and self.priority_alpha > 0:
            # 计算采样概率
            probabilities = self.priorities ** self.priority_alpha
            probabilities /= probabilities.sum()

            # 基于优先级采样
            indices = np.random.choice(
                len(self.advantages),
                size=batch_size,
                p=probabilities,
                replace=False
            )
            return indices
        else:
            # 回退到随机采样
            return super().sample_batch_indices(batch_size)


class MultiStepRolloutBuffer(CustomRolloutBuffer):
    """多步回报缓冲区"""

    def __init__(self, *args, n_steps=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """多步回报计算"""
        # 简化的多步回报实现
        gamma = 0.99
        returns = np.zeros_like(self.rewards)

        # 计算n步回报
        for step in reversed(range(len(self.rewards))):
            running_return = 0
            for t in range(step, min(step + self.n_steps, len(self.rewards))):
                running_return += (gamma ** (t - step)) * self.rewards[t]
                if dones[t]:
                    break
            returns[step] = running_return

        self.returns = returns
        self.advantages = returns - self.values.flatten()