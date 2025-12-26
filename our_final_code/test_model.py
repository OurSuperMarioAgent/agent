import time
import numpy as np
from stable_baselines3 import PPO
from env_preprocess import create_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from super_parameters import *
import torch
from algorithms import SegmentTrainingPPO

def test(
    model_path: str, 
    n_episodes: int = 20,
    total_timesteps_per_episode: int = 5000,
    fps: int = 20,
    deterministic: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    print(f"策略模式：{'确定性（最优）' if deterministic else '随机性（鲁棒性）'}")
    
    def make_test_env():
        return create_env(seed=None, eval_mode=True)
    
    env = DummyVecEnv([make_test_env])
    env = VecFrameStack(env, n_stack=n_frame_stack)
    env = VecTransposeImage(env)
    try:
        model = SegmentTrainingPPO.load(model_path, env=env, device=device)
        print(f"成功加载模型：{model_path}")
    except Exception as e:
        print(f"加载模型失败：{e}")
        return
    
    delay = 1.0 / fps  
    print(f"渲染帧率：{fps} FPS，每帧延迟：{delay:.4f} 秒")

    episode_scores = []  # 游戏原生得分（吃道具、前进、打怪等）
    episode_steps = []   # 每轮存活步数
    episode_max_x = []   # 每轮最大前进距离（Mario的x坐标）
    episode_clear = []   # 每轮是否通关（True/False）
    failure_positions = []  # 每轮失败时的x坐标
    episode_rewards = [] # 原生环境累计奖励（无自定义奖励）

    print(f"\n开始测试模型：{model_path}")
    print(f"测试轮数：{n_episodes}，每轮最大步数：{total_timesteps_per_episode}")
    print("-" * 60)

    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0  # 原生环境的累计奖励
        max_x = 0  # 本轮最大前进距离
        step = 0
        done = False
        current_x = 0  # 初始化x坐标，避免KeyError

        while not done and step < total_timesteps_per_episode:
            # 模型预测动作（控制是否用确定性策略）
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            # 累加奖励（单环境，取列表第一个元素）
            total_reward += reward[0]
            step += 1

            # 安全获取x坐标、得分等信息（避免KeyError）
            try:
                current_x = info[0].get("x_pos", 0)
                current_score = info[0].get("score", 0)
                is_clear = info[0].get("flag_get", False)
            except (IndexError, KeyError):
                current_x = 0
                current_score = 0
                is_clear = False

            # 更新最大前进距离
            if current_x > max_x:
                max_x = current_x

            # 渲染画面（测试时可视化）
            env.render()
            time.sleep(delay)
            
        # 每轮结束后暂停1秒，便于观察结果
        time.sleep(1.0)

        # 5. 记录本轮指标
        episode_scores.append(current_score)
        episode_steps.append(step)
        episode_max_x.append(max_x)
        episode_rewards.append(total_reward)
        episode_clear.append(is_clear)
        
        # 记录失败位置（未通关时）
        if not is_clear:
            failure_positions.append(current_x)

        # 打印单轮结果（格式化，更清晰）
        print(f"第{episode+1:2d}轮 | 原生奖励：{total_reward:.2f} | 游戏得分：{current_score:4d} | "
              f"步数：{step:4d} | 最大前进：{max_x:3.0f} | 通关：{'✅' if is_clear else '❌'}")

    # 6. 计算汇总指标（核心能力量化）
    env.close()
    print("-" * 60)
    print("【模型能力汇总指标】")
    print(f"1. 通关率：{sum(episode_clear)/n_episodes*100:.1f}% "
          f"（{sum(episode_clear)}/{n_episodes}轮通关）")
    print(f"2. 平均游戏得分：{np.mean(episode_scores):.0f} ± {np.std(episode_scores):.0f} "
          f"（最高：{np.max(episode_scores)}）")
    print(f"3. 平均存活步数：{np.mean(episode_steps):.0f} ± {np.std(episode_steps):.0f} "
          f"（最高：{np.max(episode_steps)}）")
    print(f"4. 平均最大前进距离：{np.mean(episode_max_x):.1f} ± {np.std(episode_max_x):.1f} "
          f"（最远：{np.max(episode_max_x)}）")
    print(f"5. 平均每步原生奖励：{np.mean(episode_rewards)/np.mean(episode_steps):.3f} "
          f"（反映动作效率）")
    
    # 统计失败位置（仅当有未通关轮次时）
    if failure_positions:
        mean_failure_pos = np.mean(failure_positions).astype(int)
        std_failure_pos = np.std(failure_positions).astype(int)
        print(f"6. 失败位置统计：平均x={mean_failure_pos} ± {std_failure_pos} "
              f"（{len(failure_positions)}轮未通关，集中在该区域）")

if __name__ == "__main__":
    test("results_and_models/final_models/final_model.zip", n_episodes=20, deterministic=False)