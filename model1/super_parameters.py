n_steps = 1024 # number of steps to run per environment per update
batch_size = 512 # batch size for each gradient update
n_episodes = 4 # number of episodes to collect per update
gamma = 0.95 # discount factor
learning_rate = 3e-4 # learning rate
ent_coef = 0.1 # exploration coefficient
clip_kl = 0.08 # target KL divergence （超过0.1的KL散度不可接受）
n_pipe = 16 # number of parallel environments
evaluate_frequency = 1e4 # evaluation frequency
resize_observation_shape = (84, 84) # resized observation shape 原图像是(240, 256)

# Frame stacking and skipping parameters
n_frame_skip = 8 # number of frames to skip