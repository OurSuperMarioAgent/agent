import gym
import numpy as np
from typing import Dict, Any, Tuple


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, use_skip=True, use_curriculum=True):
        super().__init__(env)
        self.use_skip = use_skip
        self.skip_factor = 4 if use_skip else 1
        self.use_curriculum = use_curriculum

        # 状态追踪
        self.last_x = 0
        self.max_x = 0
        self.episode_reward = 0
        self.steps_in_episode = 0
        self.consecutive_safe = 0

        # 各难点区域连续失败追踪
        self.consecutive_failures_at_1400 = 0  # 第二道沟
        self.consecutive_failures_at_1800 = 0  # 第三道沟+台阶区域
        self.consecutive_failures_at_2300 = 0  # 最后管道+飞龟区域
        self.consecutive_failures_at_2700 = 0  # 终点前跳跃区域

        # 课程学习阶段 - 细化所有关键区域
        if use_curriculum:
            # 基于1-1关卡结构的关键点位
            self.stages = [
                500,  # 第一道沟后
                1000,  # 第一个管道后
                1200,  # 第一段台阶前
                1400,  # 第二道沟
                1600,  # 第二道沟后安全区
                1800,  # 第三道沟+台阶区
                2000,  # 第三管道
                2200,  # 飞龟区域前
                2400,  # 飞龟区域后
                2600,  # 最后一个管道
                2800,  # 终点前平台
                3000  # 终点旗杆
            ]
            self.current_stage = 0
            self.stage_completed = [False] * len(self.stages)

        # 奖励参数 - 细化各区域奖励
        self.config = {
            'progress_multiplier': 0.2,
            'coin_reward': 5.0,
            'flag_reward': 1500.0,  # 提高通关奖励
            'kill_reward': 3.0,
            'time_penalty': 0.03,
            'stage_base_reward': 40.0,  # 基础奖励
            'stage_progress_bonus': 50.0,

            # 各难点区域特殊奖励
            'second_ditch_bonus': 250.0,  # 第二道沟 (1400)
            'third_ditch_bonus': 180.0,  # 第三道沟 (1800)
            'flying_turtle_bonus': 180.0,  # 飞龟区域 (2300)
            'final_jump_bonus': 250.0,  # 终点前跳跃 (2700)

            # 各区域渐进奖励系数
            'second_ditch_progress': 0.3,  # 1400-1600区域每像素奖励
            'third_ditch_progress': 0.25,  # 1800-2000区域每像素奖励
            'final_stretch_progress': 0.4,  # 2600-3000区域每像素奖励
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, original_reward, done, info = self.env.step(action)

        x_pos = info.get('x_pos', 0)
        total_reward = original_reward

        # 新增：存活步数奖励（在函数开头添加）
        self.steps_in_episode = getattr(self, 'steps_in_episode', 0) + 1
        if self.steps_in_episode > 50:  # 存活50步后开始奖励
            survival_bonus = min(0.1, self.steps_in_episode * 0.001)  # 最多0.1分/步
            total_reward += survival_bonus

        # 1. 进展奖励
        if hasattr(self, 'last_x'):
            progress = x_pos - self.last_x
            if progress > 0:
                total_reward += progress * self.config['progress_multiplier']
                if x_pos > self.max_x:
                    old_max = self.max_x
                    self.max_x = x_pos
                    breakthrough = x_pos - old_max
                    if breakthrough > 20:
                        bonus = breakthrough * 0.05
                        total_reward += bonus

                    if x_pos < 1000:
                        # 1. 存活步数奖励（随时间递增）
                        self.early_steps = getattr(self, 'early_steps', 0) + 1
                        if self.early_steps > 50:
                            survival_bonus = min(0.2, self.early_steps * 0.002)  # 最多0.2分/步
                            total_reward += survival_bonus

                        # 2. 安全前进奖励（避开敌人）
                        if progress > 0:
                            # 检测是否安全前进（无碰撞）
                            if not info.get('enemy_killed', False) and not info.get('life_decreased', False):
                                safe_progress_bonus = progress * 0.3  # 额外30%奖励
                                total_reward += safe_progress_bonus

                    # 3. 第一个沟通过特别奖励
                    if 1100 <= old_max < 1150 and x_pos >= 1150:
                        first_ditch_bonus = 150.0
                        total_reward += first_ditch_bonus

                    # ===== 区域1：第二道沟 (1400-1500) =====

                    # 修复第二道沟奖励计算
                    if 1400 <= old_max < 1500 and x_pos >= 1500:
                        total_reward += self.config['second_ditch_bonus']
                        print(f"✨ 通过第二道沟！+{self.config['second_ditch_bonus']}分")
                        if x_pos >= 1600:
                            total_reward += 100.0  # 通过后前进奖励

                    # ===== 区域2：第三道沟+台阶 (1750-1850) =====
                    if 1750 <= old_max < 1850 and x_pos >= 1850:
                        total_reward += self.config['third_ditch_bonus']
                        print(f"✨ 通过第三道沟台阶区！+{self.config['third_ditch_bonus']}分")
                        self.consecutive_failures_at_1800 = 0

                    # ===== 区域3：飞龟区域 (2250-2350) =====
                    if 2250 <= old_max < 2350 and x_pos >= 2350:
                        total_reward += self.config['flying_turtle_bonus']
                        print(f"✨ 通过飞龟区域！+{self.config['flying_turtle_bonus']}分")
                        self.consecutive_failures_at_2300 = 0

                    # ===== 区域4：终点前跳跃 (2650-2750) =====
                    if 2650 <= old_max < 2750 and x_pos >= 2750:
                        total_reward += self.config['final_jump_bonus']
                        print(f"✨ 通过终点前跳跃！+{self.config['final_jump_bonus']}分")
                        self.consecutive_failures_at_2700 = 0

        # 2. 课程学习阶段奖励
        if self.use_curriculum:
            for i, stage_target in enumerate(self.stages):
                if not self.stage_completed[i] and x_pos >= stage_target:
                    stage_reward = self.config['stage_base_reward'] * (i + 1)

                    # 关键区域额外奖励系数
                    if stage_target == 1400:  # 第二道沟
                        stage_reward *= 1.5
                    elif stage_target == 1800:  # 第三道沟
                        stage_reward *= 1.4
                    elif stage_target == 2200:  # 飞龟前
                        stage_reward *= 1.3
                    elif stage_target == 2600:  # 最后管道
                        stage_reward *= 1.6
                    elif stage_target == 2800:  # 终点前
                        stage_reward *= 1.8

                    total_reward += stage_reward
                    self.stage_completed[i] = True
                    self.current_stage = i + 1

                    # 显示所有阶段进展
                    print(f"阶段{i + 1}: {stage_target}像素 +{stage_reward:.0f}分")

        # 3. 各区域渐进奖励（持续前进奖励）
        # 第二道沟后区域 (1500-1800)
        if 1500 <= x_pos < 1800:
            progress_bonus = (x_pos - 1500) * self.config['second_ditch_progress']
            total_reward += progress_bonus

        # 第三道沟后区域 (1850-2200)
        elif 1850 <= x_pos < 2200:
            progress_bonus = (x_pos - 1850) * self.config['third_ditch_progress']
            total_reward += progress_bonus

        # 最后冲刺区域 (2600-3000)
        elif x_pos >= 2600:
            progress_bonus = (x_pos - 2600) * self.config['final_stretch_progress']
            total_reward += progress_bonus
            # 接近终点额外奖励
            if x_pos > 2800:
                proximity_bonus = (x_pos - 2800) * 0.5
                total_reward += proximity_bonus

        # 4. 金币奖励
        coins = info.get('coins', 0)
        if coins > getattr(self, 'last_coins', 0):
            coins_gained = coins - getattr(self, 'last_coins', 0)
            total_reward += coins_gained * self.config['coin_reward']

        # 5. 过关奖励
        if info.get('flag_get', False):
            flag_reward = self.config['flag_reward']
            # 根据进展给予额外奖励
            progress_factor = min(2.0, self.max_x / 3200)
            total_reward += flag_reward * progress_factor
            print(f"🎉 通关！最终奖励: {flag_reward * progress_factor:.0f}分")

        # 6. 杀敌奖励
        if info.get('enemy_killed', False):
            total_reward += self.config['kill_reward']

        # 7. 时间惩罚
        total_reward -= self.config['time_penalty']

        # 8. 死亡惩罚 - 按区域差异化
        if done and info.get('life', 3) < getattr(self, 'last_life', 3):
            death_penalty = 80.0

            # ===== 区域死亡加重惩罚 =====
            if 1300 <= x_pos <= 1500:  # 第二道沟
                death_penalty *= 1.8
                self.consecutive_failures_at_1400 += 1
                print(f"💀 第二道沟死亡！连续失败{self.consecutive_failures_at_1400}次")

            elif 1700 <= x_pos <= 1900:  # 第三道沟
                death_penalty *= 1.8
                self.consecutive_failures_at_1800 += 1
                print(f"💀 第三道沟死亡！连续失败{self.consecutive_failures_at_1800}次")

            elif 2200 <= x_pos <= 2400:  # 飞龟区域
                death_penalty *= 1.6
                self.consecutive_failures_at_2300 += 1
                print(f"💀 飞龟区域死亡！连续失败{self.consecutive_failures_at_2300}次")

            elif 2600 <= x_pos <= 2800:  # 终点前
                death_penalty *= 1.4
                self.consecutive_failures_at_2700 += 1
                print(f"💀 终点前死亡！连续失败{self.consecutive_failures_at_2700}次")

            # 进展减轻惩罚
            if self.max_x > 1500:
                progress_factor = min(0.7, self.max_x / 3200)
                death_penalty *= (1 - progress_factor)

            total_reward -= death_penalty

        # 9. 更新状态
        self.last_x = x_pos
        self.last_coins = coins
        self.last_life = info.get('life', 3)
        self.episode_reward += total_reward

        # 10. 调试信息 - 按区域分类
        if done:
            if info.get('flag_get', False):
                print(f"🎉 通关 X={x_pos} 奖励={self.episode_reward:.1f} 最远={self.max_x}")
            elif 1300 <= x_pos <= 1500:
                print(f"💀 第二道沟失败 X={x_pos} 最远={self.max_x}")
            elif 1700 <= x_pos <= 1900:
                print(f"💀 第三道沟失败 X={x_pos} 最远={self.max_x}")
            elif 2200 <= x_pos <= 2400:
                print(f"💀 飞龟区域失败 X={x_pos} 最远={self.max_x}")
            elif 2600 <= x_pos <= 2800:
                print(f"💀 终点前失败 X={x_pos} 最远={self.max_x}")
            elif x_pos > 2500:
                print(f"❌ 后期失败 X={x_pos} 最远={self.max_x}")

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_x = 0
        self.max_x = 0
        self.episode_reward = 0
        self.steps_in_episode = 0  # 重置步数计数器

        if self.use_curriculum:
            self.current_stage = 0
            self.stage_completed = [False] * len(self.stages)

        # 失败统计提醒
        failure_areas = []
        if hasattr(self, 'consecutive_failures_at_1400') and self.consecutive_failures_at_1400 > 2:
            failure_areas.append(f"第二道沟({self.consecutive_failures_at_1400}次)")
        if hasattr(self, 'consecutive_failures_at_1800') and self.consecutive_failures_at_1800 > 2:
            failure_areas.append(f"第三道沟({self.consecutive_failures_at_1800}次)")
        if hasattr(self, 'consecutive_failures_at_2300') and self.consecutive_failures_at_2300 > 2:
            failure_areas.append(f"飞龟区域({self.consecutive_failures_at_2300}次)")
        if hasattr(self, 'consecutive_failures_at_2700') and self.consecutive_failures_at_2700 > 2:
            failure_areas.append(f"终点前({self.consecutive_failures_at_2700}次)")

        if failure_areas:
            print(f"⚠️ 连续失败区域: {', '.join(failure_areas)}")

        return obs