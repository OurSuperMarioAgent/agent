# Mario agent
## 项目结构

```
mario_ppo/
├── models/                 # 训练好的模型
├── logs/                   # 训练日志
├── custom_networks/               # 所有网络相关组件
│   ├── __init__.py
│   ├── custom_cnn.py       # 自定义CNN特征提取器
│   ├── custom_mlp.py       # 自定义MLP特征提取器  
│   ├── custom_optimizers.py # 优化器和调度器
│   ├── custom_distributions.py # 动作分布
│   ├── custom_activations.py  # 激活函数
│   ├── custom_value_heads.py  # 价值函数头
│   └── __init__.py
├── buffers/                # 经验缓冲区
│   ├── custom_buffers.py
│   └── __init__.py
├── normalization/          # 标准化方法
│   ├── custom_normalize.py
│   └── __init__.py
├── callbacks/              # 回调函数
│   ├── custom_callbacks.py
│   └── __init__.py
├── rewards/                # 奖励函数
│   ├── custom_rewards.py
│   └── __init__.py
├── env_preprocess.py       # 环境预处理
├── train.py               # 训练脚本
├── super_parameters.py    # 超参数配置
└── requirements.txt
```
##
