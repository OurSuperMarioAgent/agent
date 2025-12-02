import torch
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd
from typing import Optional


class CustomVecNormalize(VecNormalize):
    """
    自定义标准化方法 - 修复版本
    兼容新版本的Stable-Baselines3
    """
    # 1. 观察标准化（像素、位置等）
    # 2. 奖励标准化（统一奖励尺度）
    # 3. 添加额外统计（最小值、最大值）
    # 4. 非线性变换（tanh处理奖励）

    def __init__(self, venv, clip_obs=10.0, clip_reward=10.0, gamma=0.99,
                 epsilon=1e-8, norm_obs=True, norm_reward=True,
                 update_obs_rms=True, update_reward_rms=True):

        # 修复：移除不兼容的参数传递
        super().__init__(
            venv,
            training=norm_obs,  # 使用training参数替代部分功能
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
            epsilon=epsilon
        )

        # 手动设置更新标志
        self.update_obs_rms = update_obs_rms
        self.update_reward_rms = update_reward_rms

        # 添加额外的统计信息
        self.obs_min = None
        self.obs_max = None

    def _update_obs_rms(self, obs: np.ndarray) -> None:
        """自定义观察标准化更新，添加范围统计"""
        if self.update_obs_rms and self.obs_rms:
            self.obs_rms.update(obs)

        # 更新观察范围统计
        if self.obs_min is None:
            self.obs_min = obs.min(axis=0)
            self.obs_max = obs.max(axis=0)
        else:
            self.obs_min = np.minimum(self.obs_min, obs.min(axis=0))
            self.obs_max = np.maximum(self.obs_max, obs.max(axis=0))

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """自定义观察标准化，添加范围归一化"""
        if self.norm_obs and self.obs_rms:
            # 基本的标准化
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

            # 添加硬裁剪
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)

        return obs

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """自定义奖励标准化"""
        if self.norm_reward and self.ret_rms:
            reward = reward / np.sqrt(self.ret_rms.var + self.epsilon)

            # 对奖励进行非线性变换
            reward = np.tanh(reward / 5.0) * 5.0  # 保持合理范围

            if self.clip_reward:
                reward = np.clip(reward, -self.clip_reward, self.clip_reward)

        return reward


class SimpleVecNormalize(VecNormalize):
    """
    简化版标准化，确保兼容性
    """

    def __init__(self, venv, **kwargs):
        # 只传递必要的参数
        super_kwargs = {
            'training': True,
            'norm_obs': kwargs.get('norm_obs', True),
            'norm_reward': kwargs.get('norm_reward', True),
            'clip_obs': kwargs.get('clip_obs', 10.0),
            'clip_reward': kwargs.get('clip_reward', 10.0),
            'gamma': kwargs.get('gamma', 0.99),
            'epsilon': kwargs.get('epsilon', 1e-8),
        }
        super().__init__(venv, **super_kwargs)