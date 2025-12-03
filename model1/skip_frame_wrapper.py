import gym

class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(SkipFrameWrapper, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info