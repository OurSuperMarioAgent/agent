import time
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from env_preprocess import create_env  # 你的环境创建函数
from stable_baselines3.common.vec_env import DummyVecEnv
from super_parameters import *  # 你的超参数（如跳帧、帧堆叠等）

def test(
    model_path: str, 
    n_episodes: int = 20,  # 测试轮数（建议≥10，统计更可靠）
    total_timesteps_per_episode: int = 5000,  # 每轮最大步数（避免无限循环）
    fps: int = 20
):
    # 1. 初始化环境和模型
    env = DummyVecEnv([create_env])
    model = PPO.load(model_path, env=env)
    delay = 2.0 / fps  # 渲染延迟（保持原逻辑）

    # 2. 定义要统计的指标（存储每轮数据）
    episode_scores = []  # 每轮最终得分（游戏内得分，含前进、吃道具等）
    episode_steps = []   # 每轮存活步数
    episode_max_x = []   # 每轮最大前进距离（Mario的x坐标，核心探索指标）
    episode_clear = []   # 每轮是否通关（True/False）
    failure_positions = []  # 每轮失败时的x坐标（找模型短板）
    episode_rewards = [] # 每轮累计奖励（训练时的奖励，和游戏得分可能不同）

    print(f"开始测试模型：{model_path}")
    print(f"测试轮数：{n_episodes}，每轮最大步数：{total_timesteps_per_episode}")
    print("-" * 50)

    # 3. 多轮测试（统计更可靠）
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0  # 训练时的累计奖励
        max_x = 0  # 本轮最大前进距离
        step = 0
        done = False

        while not done and step < total_timesteps_per_episode:
            # 模型预测动作（保持探索，deterministic=False）
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            
            # 累加奖励、更新步数
            total_reward += reward[0]  # 单环境，取列表第一个元素
            step += 1

            # 更新最大前进距离（x_pos越大，前进越远）
            current_x = info[0]["x_pos"]
            if current_x > max_x:
                max_x = current_x

            # 渲染画面（保持原逻辑）
            env.render()
            time.sleep(delay)

        # 4. 记录本轮指标
        episode_scores.append(info[0]["score"])  # 游戏内得分
        episode_steps.append(step)
        episode_max_x.append(max_x)
        episode_rewards.append(total_reward)
        # 判断是否通关（info["flag_get"]为True表示碰到旗杆）
        is_clear = info[0]["flag_get"]
        episode_clear.append(is_clear)
        # 记录失败位置（未通关时）
        if not is_clear:
            failure_positions.append(current_x)

        # 打印单轮结果
        print(f"第{episode+1:2d}轮 | 得分：{info[0]['score']:4d} | 步数：{step:4d} | "
              f"最大前进：{max_x:3.0f} | 通关：{'是' if is_clear else '否'}")

    # 5. 计算汇总指标（核心能力量化）
    env.close()
    print("-" * 50)
    print("【模型能力汇总指标】")
    print(f"1. 通关率：{sum(episode_clear)/n_episodes*100:.1f}% "
          f"（{sum(episode_clear)}/{n_episodes}轮通关）")
    print(f"2. 平均游戏得分：{np.mean(episode_scores):.0f} ± {np.std(episode_scores):.0f} "
          f"（最高：{np.max(episode_scores)}）")
    print(f"3. 平均存活步数：{np.mean(episode_steps):.0f} ± {np.std(episode_steps):.0f} "
          f"（最高：{np.max(episode_steps)}）")
    print(f"4. 平均最大前进距离：{np.mean(episode_max_x):.1f} ± {np.std(episode_max_x):.1f} "
          f"（最远：{np.max(episode_max_x)}）")
    print(f"5. 平均每步奖励：{np.mean(episode_rewards)/np.mean(episode_steps):.3f} "
          f"（反映动作效率）")
    if failure_positions:
        # 统计最常见的失败位置（取出现次数最多的前3个）
        top_failure_pos = np.round(np.mean(failure_positions)).astype(int) if len(failure_positions)>=3 else failure_positions[0]
        print(f"6. 最常见失败位置：x={top_failure_pos} 附近 "
              f"（{len(failure_positions)}轮未通关，集中在该区域）")

if __name__ == "__main__":
    # 测试20轮（可调整n_episodes）
    test("best_models/model1/best_model.zip", n_episodes=20)