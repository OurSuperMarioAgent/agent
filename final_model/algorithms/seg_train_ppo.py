from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import os

class SegmentTrainingPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_total_timesteps = None
        
    # 全局进度
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        if self.global_total_timesteps is not None:
            self._current_progress_remaining = 1.0 - float(num_timesteps) / float(self.global_total_timesteps)
        else:
            self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)
            
    # 分段learn
    def learn(
        self,
        total_timesteps: int,
        segment_timesteps: int,
        callback=None,
        log_interval: int = 1,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: str = None,
        reset_num_timesteps: bool = False,
        **kwargs
    ):
        assert total_timesteps % segment_timesteps == 0, "Total timesteps must be divisible by segment timesteps"
        num_segments = total_timesteps // segment_timesteps
        final_best_model_path = None

        self.global_total_timesteps = total_timesteps
        
        for seg in range(num_segments):
            print(f"\n=== Segment {seg+1}/{num_segments} ===")
            super().learn(
                total_timesteps=segment_timesteps,
                callback=callback,
                log_interval=log_interval,
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                tb_log_name=tb_log_name,
                eval_log_path=eval_log_path,
                reset_num_timesteps=False,
                **kwargs
            )
            final_best_model_path = self.get_best_model_path(callback)
            assert final_best_model_path is not None, "未找到历史最优模型"
            self.set_parameters(final_best_model_path)

    # 得到历史最优模型
    def get_best_model_path(self, callback):
        # 如果是CallbackList，遍历查找
        if isinstance(callback, CallbackList):
            for cb in callback.callbacks:
                if isinstance(cb, EvalCallback):
                    if hasattr(cb, "best_model_save_path") and cb.best_model_save_path is not None:
                        best_dir = cb.best_model_save_path
                        best_model_file = os.path.join(best_dir, "best_model.zip")
                        return best_model_file
        # 如果是单个callback
        elif isinstance(callback, EvalCallback):
            if hasattr(callback, "best_model_save_path") and callback.best_model_save_path is not None:
                best_dir = callback.best_model_save_path
                best_model_file = os.path.join(best_dir, "best_model.zip")
                return best_model_file
        return None
