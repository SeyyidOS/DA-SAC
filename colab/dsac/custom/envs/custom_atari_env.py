import gymnasium as gym
import numpy as np
from custom.models.dsac.sac import DSAC
from custom.spaces.custom_box import CustomBox
from gymnasium.spaces.box import Box
from stable_baselines3 import SAC
# from custom.wrappers.atari_wrapper import AtariWrapper
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from custom.models.custom_features_extractor import CustomCNN


class CustomAtariEnv(gym.Wrapper):
    def __init__(self, **kwargs):
        super(CustomAtariEnv, self).__init__(gym.make(**kwargs))

        self.action_space = Box(low=0, high=4, shape=(1,), dtype=np.float32)

    def step(self, action):
        # return self.env.step(int(np.argmax(action)))
        if type(action) == int:
            return self.env.step(action)
        return self.env.step(min(int(np.ceil(action[0])), 3))


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


if __name__ == '__main__':
    train_env, eval_env = make_env('BreakoutNoFrameskip-v4', 1, 1)

    eval_callback = EvalCallback(eval_env, best_model_save_path="/logs/",
                                 log_path="/logs/", eval_freq=100,
                                 deterministic=True, render=False)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256, attention_type='lka')
    )

    model = SAC('CnnPolicy', train_env, buffer_size=10000, learning_starts=0,
                policy_kwargs=policy_kwargs,
                verbose=0)
    model.learn(100000, progress_bar=True, callback=eval_callback)
