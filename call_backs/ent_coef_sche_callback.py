from stable_baselines3.common.callbacks import BaseCallback

class EntropyScheduleCallback(BaseCallback):
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)

    # 动态ent_coef
    def _on_step(self) -> bool:
        progress = self.model._current_progress_remaining
        if hasattr(self.model.policy, "ent_coef_schedule"):
            new_ent = self.model.policy.ent_coef_schedule(progress)
            self.model.policy.ent_coef = float(new_ent)
        return True