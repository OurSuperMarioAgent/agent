import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional, Dict, Any


class CustomCallback(BaseCallback):
    """
    自定义回调函数
    添加模型保存、性能监控、课程学习等功能
    """

    def __init__(self, check_freq: int = 1000, save_path: str = "models/",
                 eval_env=None, verbose: int = 1, early_stop_patience: int = 10):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.eval_env = eval_env
        self.early_stop_patience = early_stop_patience

        self.best_mean_reward = -np.inf
        self.best_model_path = ""
        self.patience_counter = 0
        self.episode_rewards = []

        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        """训练开始时的回调"""
        if self.verbose > 0:
            print("Training started!")

    def _on_step(self) -> bool:
        """每一步的回调"""
        # 定期检查点
        if self.n_calls % self.check_freq == 0:
            self._checkpoint()

        # 收集回合奖励
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                reward = info['episode']['r']
                self.episode_rewards.append(reward)
                if self.verbose > 0:
                    print(f"Episode reward: {reward:.2f}")

        return True

    def _checkpoint(self) -> None:
        """保存检查点和评估模型"""
        # 计算平均奖励
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])  # 最近100个回合

            # 保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.patience_counter = 0

                model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
                self.model.save(model_path)
                self.best_model_path = model_path

                if self.verbose > 0:
                    print(f"New best model saved with mean reward: {mean_reward:.2f}")
            else:
                self.patience_counter += 1

            # 提前停止检查
            if self.patience_counter >= self.early_stop_patience:
                if self.verbose > 0:
                    print(f"Early stopping triggered after {self.n_calls} steps")
                self.model.save(os.path.join(self.save_path, "final_model"))
                return False

    def _on_training_end(self) -> None:
        """训练结束时的回调"""
        if self.verbose > 0:
            print(f"Training completed! Best reward: {self.best_mean_reward:.2f}")
            if self.best_model_path:
                print(f"Best model saved at: {self.best_model_path}")


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