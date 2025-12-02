import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional, Dict, Any

# 回调函数是训练过程中的钩子（hooks），允许你在特定时间点插入自定义代码。想象成训练过程的监控摄像头+遥控器：
# 监控训练状态
# 在关键时刻执行操作
# 动态调整训练参数

class CustomCallback(BaseCallback):
    """
    更健壮的自定义回调
    """

    def __init__(self,
                 check_freq: int = 1000,
                 save_path: str = "models/",
                 eval_freq: int = 5000,
                 n_eval_episodes: int = 5,
                 early_stop_patience: int = 10,
                 verbose: int = 1):

        super().__init__(verbose)
        self.check_freq = check_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.early_stop_patience = early_stop_patience

        self.best_mean_reward = -np.inf
        self.patience_counter = 0
        self.eval_results = []

        os.makedirs(save_path, exist_ok=True)

    def _on_training_start(self):
        """训练开始时"""
        print(f"开始训练 - 总步数: {self.model._total_timesteps}")

    def _on_step(self) -> bool:
        """每一步"""
        # 定期评估
        if self.eval_env and self.n_calls % self.eval_freq == 0:
            self._evaluate_model()

        # 定期保存
        if self.n_calls % self.check_freq == 0:
            self._save_checkpoint()

        return True

    def _evaluate_model(self):
        """评估模型性能"""
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=False
        )

        self.eval_results.append({
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })

        print(f"评估结果 (步数={self.num_timesteps}): "
              f"{mean_reward:.2f} ± {std_reward:.2f}")

        # 保存最佳模型
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.patience_counter = 0

            model_path = os.path.join(
                self.save_path,
                f"best_model_{self.num_timesteps}_reward{mean_reward:.1f}"
            )
            self.model.save(model_path)
            print(f"保存最佳模型: {model_path}")
        else:
            self.patience_counter += 1
            print(f"无改进，耐心计数: {self.patience_counter}/{self.early_stop_patience}")

    def _save_checkpoint(self):
        """定期保存检查点"""
        checkpoint_path = os.path.join(
            self.save_path,
            f"checkpoint_{self.num_timesteps}"
        )
        self.model.save(checkpoint_path)
        print(f"保存检查点: {checkpoint_path}")

    def _on_training_end(self):
        """训练结束时"""
        # 保存最终模型
        final_path = os.path.join(self.save_path, "final_model")
        self.model.save(final_path)

        # 保存训练历史
        history_path = os.path.join(self.save_path, "training_history.npy")
        np.save(history_path, self.eval_results)

        print(f"训练完成!")
        print(f"最佳奖励: {self.best_mean_reward:.2f}")
        print(f"最终模型: {final_path}")



class CurriculumCallback(BaseCallback):
    """课程学习回调，逐步增加难度"""

    def __init__(self, difficulty_steps: list = [10000, 50000, 100000], verbose: int = 0):
        super().__init__(verbose)
        self.difficulty_steps = difficulty_steps
        self.current_difficulty = 0

    def _on_step(self) -> bool:
        """根据训练步数调整难度"""
        for i, step_threshold in enumerate(self.difficulty_steps):
            if self.num_timesteps >= step_threshold and self.current_difficulty <= i:
                self.current_difficulty = i + 1
                self._increase_difficulty()

        return True

    def _increase_difficulty(self):
        """增加环境难度"""
        if hasattr(self.training_env, 'env_method'):
            # 调用环境的难度增加方法
            self.training_env.env_method('increase_difficulty')

        if self.verbose > 0:
            print(f"Increased difficulty to level {self.current_difficulty} at step {self.num_timesteps}")


class LoggingCallback(BaseCallback):
    """增强的日志记录回调"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.custom_metrics = {}

    def _on_step(self) -> bool:
        """记录自定义指标"""
        # 记录价值损失
        if 'value_loss' in self.locals:
            self.logger.record('train/value_loss', self.locals['value_loss'])

        # 记录策略损失
        if 'policy_loss' in self.locals:
            self.logger.record('train/policy_loss', self.locals['policy_loss'])

        # 记录熵
        if 'entropy' in self.locals:
            self.logger.record('train/entropy', self.locals['entropy'])

        return True