import gym
import numpy as np
from typing import Dict, Any, Tuple


class CurriculumRLRewardWrapper(gym.Wrapper):
    def __init__(self, env, use_skip=True, use_curriculum=True):
        super().__init__(env)
        self.use_skip = use_skip
        self.skip_factor = 4 if use_skip else 1
        self.use_curriculum = use_curriculum

        # 状态追踪
        self.last_x = 0
        self.last_world = 1
        self.last_stage = 1
        self.max_x = 0
        self.episode_reward = 0
        self.current_world = 1
        self.current_stage = 1

        # 课程学习阶段 - 针对不同关卡设置不同的阶段
        if use_curriculum:
            # 1-1关卡阶段
            self.stages_1_1 = [500, 1000, 1500, 2000, 2500, 3000]
            # 1-2关卡阶段
            self.stages_1_2 = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
            self.current_stage_list = self.stages_1_1
            self.current_stage_idx = 0
            self.stage_completed = [False] * len(self.current_stage_list)

        # 奖励参数
        self.config = {
            'progress_multiplier': 0.5,
            'coin_reward': 5.0,
            'flag_reward': 2000.0,
            'kill_reward': 3.0,
            'time_penalty': 0.03,
            'stage_base_reward': 150.0,
            'stage_progress_bonus': 50.0,
            'world_change_bonus': 5000.0,
            'stage_change_bonus': 2000.0,
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, original_reward, done, info = self.env.step(action)

        x_pos = info.get('x_pos', 0)
        world = info.get('world', 1)
        stage = info.get('stage', 1)
        total_reward = original_reward

        # 检测关卡变化（从1-1切换到1-2）
        if world != self.current_world or stage != self.current_stage:
            print(f"关卡切换: {self.current_world}-{self.current_stage} -> {world}-{stage}")

            if (self.current_world == 1 and self.current_stage == 1 and
                    world == 1 and stage == 2):
                # 1-1通关，进入1-2
                total_reward += self.config['stage_change_bonus']
                print(f"进入1-2关卡 +{self.config['stage_change_bonus']}分")

                # 重置课程学习阶段为1-2的阶段
                if self.use_curriculum:
                    self.current_stage_list = self.stages_1_2
                    self.current_stage_idx = 0
                    self.stage_completed = [False] * len(self.current_stage_list)
                    print("已切换到1-2关卡阶段目标")

            self.current_world = world
            self.current_stage = stage
            # 重置位置追踪
            self.last_x = x_pos
            self.max_x = x_pos

        # 1. 进展奖励
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

        # 2. 课程学习阶段奖励（根据当前关卡使用对应的阶段）
        if self.use_curriculum:
            # 根据当前关卡选择阶段列表
            stage_list = self.stages_1_2 if self.current_stage == 2 else self.stages_1_1

            for i, stage_target in enumerate(stage_list):
                if i < len(self.stage_completed) and not self.stage_completed[i] and x_pos >= stage_target:
                    if self.current_stage == 2:
                        stage_reward = (self.config['stage_base_reward'] + 100.0) * 0.5 * (i + 1) *(i + 1) + self.config['stage_base_reward'] * 6
                        if i == 7:
                            stage_reward += 1500.0
                    else:
                        stage_reward = self.config['stage_base_reward'] * (i + 1)
                    # 1-2关卡有额外奖励
                    # if self.current_stage == 2:
                    #     stage_reward *= 1.2

                    total_reward += stage_reward
                    self.stage_completed[i] = True
                    self.current_stage_idx = i + 1

                    # 只显示重要阶段
                    if stage_target >= 2000 or self.current_stage == 2:
                        stage_info = f"1-{self.current_stage}" if self.current_stage == 2 else f"{stage_target}像素"
                        print(f"阶段{i + 1}: {stage_info} +{stage_reward:.0f}分")

        # 3. 接近终点奖励（根据关卡调整终点位置）
        if self.current_stage == 2:
            # 1-2关卡：终点约4000像素
            target_distance = 4000
            if x_pos > target_distance - 500:  # 距离终点500像素开始奖励
                proximity_bonus = (x_pos - (target_distance - 500)) * 0.1
                total_reward += proximity_bonus
                if x_pos > target_distance - 200 and self.last_x <= target_distance - 200:
                    total_reward += 150.0  # 1-2关卡更高的接近奖励
        else:
            # 1-1关卡：终点约3000像素
            if x_pos > 2500:
                proximity_bonus = (x_pos - 2500) * 0.1
                total_reward += proximity_bonus
                if x_pos > 2800 and self.last_x <= 2800:
                    total_reward += 100.0

        # 4. 金币奖励
        coins = info.get('coins', 0)
        if coins > getattr(self, 'last_coins', 0):
            coins_gained = coins - getattr(self, 'last_coins', 0)
            total_reward += coins_gained * self.config['coin_reward']

        # 5. 过关奖励
        # if info.get('flag_get', False):
        #     total_reward += self.config['flag_reward']
        #     world_stage = f"{world}-{stage}"
        #     print(f"通关{world_stage} +{self.config['flag_reward']}分")

        # 6. 杀敌奖励
        if info.get('enemy_killed', False):
            total_reward += self.config['kill_reward']

        # 7. 时间惩罚
        total_reward -= self.config['time_penalty']

        # 8. 死亡惩罚（根据当前关卡调整）
        if done and info.get('life', 3) < getattr(self, 'last_life', 3):
            death_penalty = 50.0

            # 根据关卡进度调整惩罚
            if self.current_stage == 2:
                # 1-2关卡：基于4000像素的总长度
                progress_factor = min(0.7, self.max_x / 4000)
            else:
                # 1-1关卡：基于3200像素的总长度
                progress_factor = min(0.7, self.max_x / 3200)

            death_penalty *= (1 - progress_factor)
            total_reward -= death_penalty

        # 9. 更新状态
        self.last_x = x_pos
        self.last_coins = coins
        self.last_life = info.get('life', 3)
        self.episode_reward += total_reward

        # 10. 调试信息
        if done:
            world_stage = f"{world}-{stage}"
            if info.get('flag_get', False):
                print(f"通关{world_stage} X={x_pos} 奖励={self.episode_reward:.1f}")
            elif x_pos > (4000 if stage == 2 else 2500):
                print(f"失败{world_stage} X={x_pos} 最远={self.max_x}")

        return obs, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_x = 0
        self.max_x = 0
        self.episode_reward = 0
        self.current_world = 1
        self.current_stage = 1

        if self.use_curriculum:
            # 默认从1-1阶段开始
            self.current_stage_list = self.stages_1_1
            self.current_stage_idx = 0
            self.stage_completed = [False] * len(self.current_stage_list)

        return obs