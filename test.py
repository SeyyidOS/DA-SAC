import gym
from stable_baselines3.common.evaluation import evaluate_policy
from src.utils import ALGOS


def model_test(env_name, model_path, algo):
    env = gym.make(env_name, render_mode="human")

    model = ALGOS[algo].load(model_path, env=env)

    ## Evaluate the agent
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
    # print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Optionally, visualize the agent's performance
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        test_steps = 0
        total_reward = 0
        while not done and test_steps < 200:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, truncated, info = env.step(action)
            env.render()
            total_reward += rewards
            test_steps += 1
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    env.close()


if __name__ == '__main__':
    model_test(
        env_name='MountainCarContinuous-v0',
        model_path='logs/sac/MountainCarContinuous-v0_18/best_model.zip',
        algo='sac'
    )
