import math

class CosineAnnealingSchedule:
    """余弦退火学习率调度器"""

    def __init__(self, initial_lr=3e-4, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.min_lr = min_lr

    def __call__(self, progress_remaining):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
    
def lr_schedule(p):
        lr_scheduler = CosineAnnealingSchedule(
            initial_lr=3e-4,
            min_lr=1e-6
        )
        return lr_scheduler(p)