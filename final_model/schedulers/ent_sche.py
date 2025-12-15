from .lr_sche import CosineAnnealingSchedule

def ent_schedule(p):
        # p 是剩余进度 (从1到0)
        # 使用你的CosineAnnealingSchedule
        ent_scheduler = CosineAnnealingSchedule(
            initial_lr=2e-2,
            min_lr=1e-4
        )
        return ent_scheduler(p)