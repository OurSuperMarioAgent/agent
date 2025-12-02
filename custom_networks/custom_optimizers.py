import torch
import math
from torch.optim import AdamW, Optimizer

#原文：(PPO->ActorCriticPolicy->__init__,PPO->ActorCriticCnnPolicy->__init__)
#optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam

class CustomOptimizer(AdamW):
    def __init__(self, params, lr=1e-3, max_grad_norm=0.5, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.max_grad_norm = max_grad_norm

    def step(self, closure=None):
        # 梯度裁剪，防止梯度爆炸
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        super().step(closure)


class CosineAnnealingSchedule:
    """余弦退火学习率调度器"""
    # 平滑过渡：学习率从最大值平滑下降到最小值
    # 探索与利用的平衡：初期：大学习率快速探索；后期：小学习率精细调参

    def __init__(self, initial_lr=3e-4, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.min_lr = min_lr

    def __call__(self, progress_remaining):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay