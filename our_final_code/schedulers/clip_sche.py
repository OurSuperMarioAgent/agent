from .lr_sche import CosineAnnealingSchedule

def clip_schedule(p):
        clip_scheduler = CosineAnnealingSchedule(
            initial_lr=0.2,
            min_lr=0.05
        )
        return clip_scheduler(p)