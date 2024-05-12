import gym
from gym import spaces
import numpy as np
from src.models.dsac.sac import DSAC

class CustomAtariEnv(gym.Wrapper):
    def __init__(self, **kwargs):
        super(CustomAtariEnv, self).__init__(gym.make(**kwargs))
        # Custom action space, example continuous

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        # Convert continuous action to discrete action
        # Assuming the action space originally had N discrete actions
        discrete_actions = self.env.unwrapped.get_action_meanings()
        num_discrete_actions = len(discrete_actions)

        # Simple example: Map continuous actions to discrete based on some criterion
        # This is just a placeholder for a proper mapping based on your specific needs
        discrete_action = int((action[0] + 1) / 2 * (num_discrete_actions - 1))
        discrete_action = np.clip(discrete_action, 0, num_discrete_actions - 1)

        return self.env.step(discrete_action)


env = CustomAtariEnv(id='Breakout-v0')

model = DSAC('MlpPolicy', env, buffer_size=10000, verbose=1)
model.learn(100000)
# Usage
# obs = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()  # Sample a random continuous action
#     action = model(obs)
#     obs, reward, done, info, _ = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
