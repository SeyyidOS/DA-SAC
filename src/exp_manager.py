import yaml
import os
import gym
import uuid

from .utils import ALGOS, get_latest_run_id
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, ProgressBarCallback
import sys
from typing import Any, Dict, Tuple, Union

import mlflow
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
            self,
            key_values: Dict[str, Any],
            key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
            step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
                sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


class ExperimentManager:
    def __init__(self, args):
        self.args = args

        # uuid_str = f"_{uuid.uuid4()}"
        self.log_path = f"{args.log_folder}/{args.algo}/"
        self.save_path = os.path.join(
            self.log_path, f"{self.args.env}_{get_latest_run_id(self.log_path, args.env) + 1}"
        )
        self.params_path = f"{self.save_path}/{args.env}"

        self.callbacks = []

    def setup_experiment(self):
        hyperparams = self.read_hyperparameters()

        self.create_log_folder()
        self.create_callbacks()

        n_envs = 1 if self.args.optimize else self.args.n_envs
        env = self.create_envs(n_envs, no_log=False)

        if self.args.pretrained_agent:
            model = self._load_pretrained_agent(hyperparams, env)
        elif self.args.optimize:
            env.close()
            return None
        else:
            # Train an agent from scratch
            model = ALGOS[self.args.algo](
                env=env,
                seed=self.args.seed,
                verbose=self.args.verbose,
                device=self.args.device,
                **hyperparams,
            )
        return model, hyperparams

    def read_hyperparameters(self):
        with open(self.args.conf) as f:
            hyperparams_dict = yaml.safe_load(f)

        if self.args.env in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[self.args.env]
        else:
            raise ValueError(
                f"Hyperparameters not found for {self.args.algo}-{self.args.env} in {self.args.conf}"
            )

        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
            if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

        return hyperparams

    def create_log_folder(self):
        os.makedirs(self.params_path, exist_ok=True)

    def create_envs(self, n_envs: int, eval_env: bool = False, no_log: bool = False) -> VecEnv:

        log_dir = None if eval_env or no_log else self.save_path

        spec = gym.spec(self.args.env)

        def make_env(**kwargs) -> gym.Env:
            return spec.make(**kwargs)

        env = make_vec_env(
            make_env,
            n_envs=n_envs,
            seed=self.args.seed,
            monitor_dir=log_dir,
        )

        return env

    def create_callbacks(self):
        if self.args.verbose:
            self.callbacks.append(ProgressBarCallback())

        if self.args.eval_freq > 0 and not self.args.optimize:
            # Account for the number of parallel environments
            self.args.eval_freq = max(self.args.eval_freq // self.args.n_envs, 1)

            if self.args.verbose > 0:
                print("Creating test environment")

            # save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
            eval_callback = EvalCallback(
                self.create_envs(self.args.n_eval_envs, eval_env=True),
                # callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.save_path,
                n_eval_episodes=self.args.eval_episodes,
                log_path=self.save_path,
                eval_freq=self.args.eval_freq,
                deterministic=self.args.deterministic_eval,
            )

            self.callbacks.append(eval_callback)

    def learn(self, model: BaseAlgorithm) -> None:
        """
        :param model: an initialized RL model
        """
        kwargs: Dict[str, Any] = {}

        if self.args.log_interval > -1:
            kwargs = {"log_interval": self.args.log_interval}

        kwargs["callback"] = self.callbacks

        if self.args.track:
            loggers = Logger(
                folder=None,
                output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
            )
            model.set_logger(loggers)

        model.learn(self.args.n_timesteps, **kwargs)

    def save_trained_model(self, model: BaseAlgorithm) -> None:
        """
        Save trained model optionally with its replay buffer
        and ``VecNormalize`` statistics

        :param model:
        """
        print(f"Saving to {self.save_path}")
        model.save(f"{self.save_path}/{self.args.env}")

    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            verbose=self.verbose,
            device=self.device,
            **hyperparams,
        )

    def objective(self):
        pass

    def hyperparameters_optimization(self):
        pass
