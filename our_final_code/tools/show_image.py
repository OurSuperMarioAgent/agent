import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def show_original_mario_image():
    # 1. 创建原始 Mario 环境（无任何包装器，保留原生图像）
    # 注意：JoypadSpace 仅简化动作空间，不修改图像，可保留（否则环境动作空间过大无法 reset）
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)  # 必须加，否则原生动作空间会报错
    
    # 2. 重置环境，获取原始观测（RGB 图像）
    obs = env.reset()
    
    # 3. 打印原始图像关键信息（帮助确定裁剪区域）
    print("=" * 60)
    print("原始 Mario 图像信息：")
    print(f"图像形状（H, W, C）：{obs.shape} → 高度(H)={obs.shape[0]}, 宽度(W)={obs.shape[1]}, 通道数(C)={obs.shape[2]}")
    print(f"像素值范围：{obs.min()} ~ {obs.max()}（RGB 图像，0-255 正常）")
    print(f"数据类型：{obs.dtype}")
    print("=" * 60)
    print("提示：")
    print("- 高度 240 像素：0（顶部）→ 240（底部）")
    print("- 宽度 256 像素：0（左侧）→ 256（右侧）")
    print("- 通常顶部 20 像素是分数栏，底部 20 像素是冗余边框，可裁剪")
    print("=" * 60)
    
    # 4. 显示原始图像（带网格，方便看像素坐标）
    plt.figure(figsize=(8, 6))
    plt.imshow(obs)  # matplotlib 直接支持 RGB 格式 (H, W, 3)
    plt.title("原始 Mario 图像（带像素网格）", fontsize=12)
    plt.xlabel("宽度 (W) 像素坐标（0→256）", fontsize=10)
    plt.ylabel("高度 (H) 像素坐标（0→240）", fontsize=10)
    plt.grid(True, alpha=0.3, color='red')  # 红色网格，方便定位像素
    plt.xticks(np.arange(0, 257, 32))  # 宽度每 32 像素标刻度
    plt.yticks(np.arange(0, 241, 32))  # 高度每 32 像素标刻度
    
    # 5. 保存图像到本地（可选，方便后续查看）
    plt.savefig("mario_original_image.png", dpi=150, bbox_inches='tight')
    print("原始图像已保存为：mario_original_image.png")
    
    # 6. 显示图像窗口（阻塞，关闭窗口后继续执行）
    plt.show()
    
    # 7. 关闭环境，释放资源
    env.close()

if __name__ == "__main__":
    show_original_mario_image()