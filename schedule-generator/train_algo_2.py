from tqdm import tqdm
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class TrainAlgo2:
    def __init__(self, agent, env):
        self.env = gym.wrappers.RecordEpisodeStatistics(env) #, deque_size=n_episodes)
        self.agent = agent

    def train(self, n_episodes):
        agent = self.agent
        env = self.env

        #env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
        for episode in tqdm(range(n_episodes)):
            obs, info = env.reset()
            done = False

            # play one episode
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # update the agent
                agent.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            agent.decay_epsilon()

    def plot(self):   
        env = self.env
        agent = self.agent
            
        rolling_length = 500
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        axs[0].set_title("Episode rewards")
        reward_moving_average = (
            np.convolve(
                np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[1].set_title("Episode lengths")
        length_moving_average = (
            np.convolve(
                np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[2].set_title("Training Error")
        training_error_moving_average = (
            np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
            / rolling_length
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        plt.tight_layout()
        plt.show()