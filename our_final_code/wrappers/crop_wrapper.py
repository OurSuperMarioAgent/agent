import gym
import numpy as np
from gym import ObservationWrapper
from gym.spaces import Box

class CropObservation(ObservationWrapper):
    """
    Gym 观测包装器：裁剪图像的指定区域
    
    :param env: 被包装的 Gym 环境实例
    :param crop_box: 裁剪区域四元组 (top, left, bottom, right)
                     - top: 裁剪起始行（从上往下数，像素坐标）
                     - left: 裁剪起始列（从左往右数，像素坐标）
                     - bottom: 裁剪结束行（不包含，像素坐标）
                     - right: 裁剪结束列（不包含，像素坐标）
    """
    def __init__(self, env: gym.Env, crop_box: tuple):
        super().__init__(env)
        
        # 1. 验证裁剪框的合法性
        assert len(crop_box) == 4, f"crop_box 必须是 (top, left, bottom, right) 四元组，当前输入：{crop_box}"
        self.top, self.left, self.bottom, self.right = crop_box
        assert self.top < self.bottom, f"top ({self.top}) 必须小于 bottom ({self.bottom})"
        assert self.left < self.right, f"left ({self.left}) 必须小于 right ({self.right})"
        
        # 2. 获取原始观测空间的形状和类型
        original_shape = self.observation_space.shape
        original_dtype = self.observation_space.dtype
        original_low = self.observation_space.low
        original_high = self.observation_space.high
        
        # 3. 计算裁剪后的观测形状（兼容 2D 灰度图/3D RGB 图）
        if len(original_shape) == 3:
            new_height = self.bottom - self.top
            new_width = self.right - self.left
            new_channels = original_shape[2]
            self.cropped_shape = (new_height, new_width, new_channels)
        elif len(original_shape) == 2:
            new_height = self.bottom - self.top
            new_width = self.right - self.left
            self.cropped_shape = (new_height, new_width)
        else:
            raise ValueError(f"不支持的观测形状：{original_shape}，仅支持 2D/3D 图像")
        
        # 4. 更新观测空间
        self.observation_space = Box(
            low=original_low.min(),
            high=original_high.max(),
            shape=self.cropped_shape,
            dtype=original_dtype
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        对每一步的观测执行裁剪操作（核心方法）
        :param obs: 原始观测（numpy 数组，形状 (H, W) 或 (H, W, C)）
        :return: 裁剪后的观测
        """
        cropped_obs = obs[self.top:self.bottom, self.left:self.right, ...]
        return cropped_obs