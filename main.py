from stable_baselines3.common.callbacks import EvalCallback
from src.models.custom_features_extractor import CustomCNN
from src.envs.custom_atari_env import make_env
from src.optimization.optimze import optimize
from stable_baselines3 import SAC


def train():
    train_env, eval_env = make_env('BreakoutNoFrameskip-v4', 1, 1)

    eval_callback = EvalCallback(eval_env, best_model_save_path="/logs/",
                                 log_path="/logs/", eval_freq=100,
                                 deterministic=True, render=False)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256, attention_type='csa')
    )

    model = SAC('CnnPolicy', train_env, buffer_size=10000, learning_starts=0,
                policy_kwargs=policy_kwargs,
                verbose=0)
    model.learn(100000, progress_bar=True, callback=eval_callback)


if __name__ == '__main__':
    # train()
    optimize()
