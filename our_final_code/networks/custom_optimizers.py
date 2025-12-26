import torch
import math
from torch.optim import AdamW, Optimizer

class CustomOptimizer(AdamW):
    def __init__(self, params, lr=1e-3, max_grad_norm=0.5, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.max_grad_norm = max_grad_norm

    def step(self, closure=None):
        # 梯度裁剪，防止梯度爆炸
        if self.max_grad_norm > 0:
            # 收集所有参数
            parameters = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        parameters.append(p)

            # 裁剪梯度
            if parameters:
                torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

        super().step(closure)