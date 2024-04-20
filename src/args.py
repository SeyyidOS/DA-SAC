import argparse
from .exp_manager import ExperimentManager
from .utils import ALGOS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", default="ppo", type=str, choices=list(ALGOS.keys()), help="RL Algorithm"
    )
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="environment ID"
    )
    parser.add_argument(
        "--deterministic-eval", default=False, action="store_true", help="Run hyperparameters search"
    )
    parser.add_argument(
        "--optimize", default=False, action="store_true", help="Run hyperparameters search"
    )
    parser.add_argument(
        "--track", default=False, action="store_true", help="Track trainig using mlflow"
    )
    parser.add_argument(
        "--conf", default=None, type=str,
        help="Custom yaml file or python package from which the hyperparameters will be loaded."
    )
    parser.add_argument(
        "--device", default="auto", type=str,
        help="PyTorch device to be use (ex: cpu, cuda...)"
    )
    parser.add_argument(
        "--verbose", default=1, type=int,
        help="Verbose mode (0: no output, 1: INFO)",
    )
    parser.add_argument("--seed", default=-1, type=int,  help="Random generator seed",)
    parser.add_argument(
        "-n", "--n-timesteps",  default=-1, type=int, help="Overwrite the number of timesteps"
    )
    parser.add_argument("-f", "--log-folder", default="logs", type=str, help="Log folder")
    parser.add_argument(
        "--n-envs",  default=1, type=int, help="Overwrite the number of environments"
    )
    parser.add_argument(
        "--n-eval-envs",  default=1, type=int, help="Number of evaluation environments"
    )
    parser.add_argument(
        "--eval-episodes",  default=5, type=int, help="Number of episodes to use for evaluation"
    )
    parser.add_argument(
        "--save-freq", default=-1, type=int, help="Save the model every n steps (if negative, no checkpoint)"
    )
    parser.add_argument(
        "--eval-freq", default=25000, type=int,
        help="Evaluate the agent every n steps (if negative, no evaluation). "
    )
    parser.add_argument(
        "--log-interval", default=-1, type=int, help="Override log interval (default: -1, no change)"
    )
    parser.add_argument(
        "--pretrained-agent",  default=None, type=str, help="Pretrained Agent Path"
    )
    parser.add_argument(
        "--tracking-uri",  default='http://127.0.0.1:8080/', type=str, help="Pretrained Agent Path"
    )
    parser.add_argument(
            "--experiment-name",  default='Temporary Experiments', type=str, help="Pretrained Agent Path"
    )
    args = parser.parse_args()

    exp_manager = ExperimentManager(args)

    return args, exp_manager
