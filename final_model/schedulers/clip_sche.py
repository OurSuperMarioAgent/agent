from .lr_sche import CosineAnnealingSchedule

def clip_schedule(p):
        # p 是剩余进度 (从1到0)
        # 使用你的CosineAnnealingSchedule
        clip_scheduler = CosineAnnealingSchedule(
            initial_lr=0.2,
            min_lr=0.05
        )
        return clip_scheduler(p)