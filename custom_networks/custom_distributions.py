import torch
from stable_baselines3.common.distributions import CategoricalDistribution

class CustomDistribution(CategoricalDistribution):
    def __init__(self, action_dim, initial_beta=0.01, decay_rate=0.9995):
        super().__init__(action_dim)
        self.beta = initial_beta  # 熵系数
        self.decay_rate = decay_rate
        self.train_step = 0

    def entropy(self):
        """带自适应系数的熵计算"""
        base_entropy = super().entropy()
        # 随着训练衰减熵系数
        self.beta *= self.decay_rate
        self.train_step += 1
        return base_entropy * self.beta

    def update_beta(self, performance_metric):
        """根据性能动态调整熵系数"""
        # 如果性能提升慢，增加探索
        # 动态调整：理想情况是训练初期熵高（多探索），后期熵低（多利用）
        if performance_metric < 0.1:  # 性能差
            self.beta = min(self.beta * 1.1, 0.1)  # 增加探索
        else:  # 性能好
            self.beta = max(self.beta * 0.99, 0.001)  # 减少探索


class ProjectDistribution(CategoricalDistribution):
    """
    项目级改进：自适应熵调整 + 探索奖励
    核心思想：根据训练表现动态调整探索程度
    """
    def __init__(self, action_dim):
        super().__init__(action_dim)

        # 记录训练状态
        self.entropy_history = []  # 记录最近100步的熵值
        self.convergence_counter = 0  # 收敛计数器

        # 可调参数
        self.min_entropy = 0.1  # 最小熵（保证基本探索）
        self.max_entropy = 2.0  # 最大熵（防止过度随机）
        self.convergence_threshold = 0.05  # 收敛判断阈值

    def entropy(self):
        """改进的熵计算：防止过早收敛"""
        # 1. 计算基础熵
        base_entropy = super().entropy()

        # 2. 记录历史用于分析
        self._record_entropy(base_entropy.item())

        # 3. 检测是否过早收敛（熵持续很低）
        if self._is_premature_convergence():
            # 如果过早收敛，人为提高熵鼓励探索
            boosted_entropy = base_entropy * 2.0
        else:
            boosted_entropy = base_entropy

        # 4. 裁剪到合理范围
        final_entropy = torch.clamp(boosted_entropy, self.min_entropy, self.max_entropy)

        return final_entropy

    def log_prob(self, actions):
        """标准log概率计算，保持兼容性"""
        return super().log_prob(actions)

    def sample(self):
        """标准采样"""
        return super().sample()

    def _record_entropy(self, entropy_value):
        """记录熵历史"""
        self.entropy_history.append(entropy_value)
        if len(self.entropy_history) > 100:  # 只保留最近100个
            self.entropy_history.pop(0)

    def _is_premature_convergence(self):
        """检测是否过早收敛"""
        if len(self.entropy_history) < 20:
            return False

        # 计算最近20步的平均熵
        recent_entropy = sum(self.entropy_history[-20:]) / 20

        # 如果熵持续很低，说明可能过早收敛
        if recent_entropy < self.convergence_threshold:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0

        # 连续10次检测到低熵才算过早收敛
        return self.convergence_counter > 10