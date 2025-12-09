import torch
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd
from typing import Optional


class SimpleVecNormalize(VecNormalize):
    """只添加范围统计，不破坏原有逻辑"""

    def __init__(self, venv, **kwargs):
        super().__init__(venv, **kwargs)
        self.obs_min = None
        self.obs_max = None

    def step_wait(self):
        obs, rews, dones, infos = super().step_wait()

        # 更新统计但不影响标准化
        if self.obs_min is None:
            self.obs_min = obs.min(axis=0)
            self.obs_max = obs.max(axis=0)
        else:
            self.obs_min = np.minimum(self.obs_min, obs.min(axis=0))
            self.obs_max = np.maximum(self.obs_max, obs.max(axis=0))

        return obs, rews, dones, infos

    def get_obs_stats(self):
        """获取观测统计信息"""
        return {
            'min': self.obs_min,
            'max': self.obs_max,
            'mean': self.obs_rms.mean if self.obs_rms else None,
            'var': self.obs_rms.var if self.obs_rms else None
        }

class VecNormalize(VecNormalize):
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