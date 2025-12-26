import gym_super_mario_bros
import pygame
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# 修正后的按键映射：明确 A=跳跃（空格），B=加速（X）
KEY_MAPPING = {
    "right": pygame.K_RIGHT,
    "left": pygame.K_LEFT,
    "jump": pygame.K_SPACE,  # A 键 → 跳跃
    "speed": pygame.K_x      # B 键 → 加速（需配合方向键）
}

def manual_play_mario():
    # 1. 初始化环境
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    obs = env.reset()
    
    # 2. 初始化 pygame 窗口
    original_h, original_w = obs.shape[0], obs.shape[1]
    scale = 2
    screen_w, screen_h = original_w * scale, original_h * scale
    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Mario 手动模式 | 空格=跳 | X+右键=加速跑 | ESC=退出")
    
    # 3. 帧率控制
    clock = pygame.time.Clock()
    fps = 30
    
    # 打印操作提示
    print("=" * 80)
    print("✅ 修正后的操作说明：")
    print("→ 右箭头：向右走")
    print("→ 右箭头 + 空格：向右跳（A键）")
    print("→ 右箭头 + X 键：向右加速跑（B键）")
    print("→ 右箭头 + X + 空格：向右跳+加速（A+B键）")
    print("→ 左箭头：向左走")
    print("→ 空格：原地跳")
    print("→ ESC 键 / 关闭窗口：退出游戏")
    print("⚠ 注意：X键（加速）必须和方向键配合才有效，单独按X无动作！")
    print("=" * 80)
    
    # 4. 主循环
    running = True
    while running:
        action = 0  # 默认无操作
        keys = pygame.key.get_pressed()
        
        # 处理退出
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # 修正后的动作映射逻辑（优先级：跳+加速 > 加速 > 跳 > 纯方向）
        if keys[KEY_MAPPING["right"]]:
            if keys[KEY_MAPPING["jump"]] and keys[KEY_MAPPING["speed"]]:
                action = 4  # 向右跳+加速（right A B）
            elif keys[KEY_MAPPING["speed"]]:
                action = 3  # 向右加速跑（right B）← 修正核心：这里改为3
            elif keys[KEY_MAPPING["jump"]]:
                action = 2  # 向右跳（right A）
            else:
                action = 1  # 向右走（right）
        elif keys[KEY_MAPPING["left"]]:
            action = 6  # 向左走
        elif keys[KEY_MAPPING["jump"]]:
            action = 5  # 原地跳
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
            print("游戏结束，已重置！")
        
        # 渲染画面
        frame = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        frame = pygame.transform.scale(frame, (screen_w, screen_h))
        screen.blit(frame, (0, 0))
        
        # 绘制操作提示文字
        font = pygame.font.SysFont(None, 22)
        hint_text = font.render("X+右键=加速跑 | 空格=跳 | ESC=退出", True, (255, 255, 0))
        screen.blit(hint_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(fps)
    
    # 清理资源
    env.close()
    pygame.quit()
    print("游戏退出！")

if __name__ == "__main__":
    manual_play_mario()