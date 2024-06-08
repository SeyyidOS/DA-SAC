from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from stable_baselines3.common.callbacks import EvalCallback
from src.models.custom_features_extractor import CustomCNN
from src.envs.custom_atari_env import make_env
from src.optimization.sampler import sample_sac_params
from stable_baselines3 import SAC
from datetime import datetime

import optuna


def objective_function(trial: optuna.Trial, env, n_env, n_timesteps, eval_freq, attention_type):
    sampled_hyperparams = sample_sac_params(trial)

    train_env, eval_env = make_env(env, n_env, 1)

    eval_callback = EvalCallback(eval_env, best_model_save_path="/logs/",
                                 log_path="/logs/", eval_freq=eval_freq,
                                 deterministic=True, render=False)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=sampled_hyperparams['features_dim'], attention_type=attention_type,
                                       conv_channels=sampled_hyperparams['conv_channels'],
                                       conv_kernel_size=sampled_hyperparams['conv_kernel_size'])
    )

    sampled_hyperparams['policy_kwargs'].update(policy_kwargs)

    del sampled_hyperparams['features_dim']
    del sampled_hyperparams['conv_channels']
    del sampled_hyperparams['conv_kernel_size']

    model = SAC('CnnPolicy', train_env,
                verbose=0,
                **sampled_hyperparams)
    model.learn(n_timesteps, progress_bar=True, callback=eval_callback)

    reward = eval_callback.last_mean_reward

    return reward


def optimize(env='BreakoutNoFrameskip-v4', n_env=5, n_timesteps=100, eval_freq=100, attention_type='csa'):
    def _create_sampler(sampler_method: str):
        # n_warmup_steps: Disable pruner until the trial reaches the given number of steps.
        if sampler_method == "random":
            sampler: BaseSampler = RandomSampler(seed=42)
        elif sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=0, seed=42, multivariate=True)
        elif sampler_method == "skopt":
            from optuna.integration.skopt import SkoptSampler
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler

    def _create_pruner(pruner_method: str) -> BasePruner:
        if pruner_method == "halving":
            pruner: BasePruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=0, n_warmup_steps=n_timesteps // eval_freq // 3)
        elif pruner_method == "none":
            # Do not prune
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner

    study = optuna.create_study(
        sampler=_create_sampler('tpe'),
        pruner=_create_pruner('median'),
        direction="maximize",
    )

    try:
        study.optimize(lambda trial: objective_function(trial, env, n_env, n_timesteps, eval_freq, attention_type),
                       n_trials=10)
    except:
        pass

    folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    study.trials_dataframe().to_csv(f"optimization_log_{env}_{attention_type}_{folder_name}.csv")


if __name__ == '__main__':
    optimize()
