from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


class ProgressBarCallback(BaseCallback):
    """
    Custom callback for plotting the training progress.
    """

    def __init__(self, l, g):
        super(ProgressBarCallback, self).__init__()
        self.total_timesteps = l['self'].num_timesteps
        self.pbar = None

    def _on_training_start(self):
        """
        This method is called before the first rollout starts.
        """
        self.pbar = tqdm(total=self.total_timesteps, desc='Training Progress')

    def _on_step(self):
        """
        This method is called by the model after each call to `env.step()`.
        """
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        """
        This method is called by the model after the training is completed.
        """
        self.pbar.close()
