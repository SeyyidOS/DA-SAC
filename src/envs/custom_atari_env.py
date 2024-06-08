from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback
from src.models.custom_features_extractor import CustomCNN
from stable_baselines3.common.monitor import Monitor
from gymnasium.spaces.box import Box
from stable_baselines3 import SAC

import gymnasium as gym
import numpy as np


class CustomAtariEnv(gym.Wrapper):
    def __init__(self, **kwargs):
        super(CustomAtariEnv, self).__init__(gym.make(**kwargs))
        self.n = self.action_space.n
        self.action_space = Box(low=0, high=self.n, shape=(1,), dtype=np.float32)

    def step(self, action):
        # return self.env.step(int(np.argmax(action)))
        if type(action) == int:
            return self.env.step(action)
        return self.env.step(min(int(np.ceil(action[0])), self.n - 1))


def make_env(id, n_train_envs, n_eval_envs):
    def _make_env():
        env = CustomAtariEnv(id=id)
        env = Monitor(env, filename="/logs/")
        env = AtariWrapper(env)
        return env

    train_env = DummyVecEnv([_make_env for i in range(n_train_envs)])
    eval_env = DummyVecEnv([_make_env for i in range(n_eval_envs)])

    train_env = VecTransposeImage(train_env)
    eval_env = VecTransposeImage(eval_env)

    return train_env, eval_env



