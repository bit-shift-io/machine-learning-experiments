from __future__ import annotations
from collections import defaultdict
import numpy as np
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten

# hyperparameters
learning_rate = 0.01
n_episodes = 50
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

class QLearnAgent:
    def __init__(
        self,
        env,
        learning_rate: float = 0.01,
        initial_epsilon: float = start_epsilon,
        epsilon_decay: float = epsilon_decay,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env

        self.q_values = defaultdict(lambda: {}) #np.zeros(env.n_actions))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            obs_hash = tuple(flatten(self.env.observation_space, obs))
            q_values = self.q_values[obs_hash]

            # if we havenn't explored this state yet, return a random sample
            if len(q_values) <= 0:
                return self.env.action_space.sample()

            q_value = max(q_values, key=q_values.get)
            action = unflatten(self.env.action_space, list(q_value))
            return action #int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        next_obs_hash = tuple(flatten(self.env.observation_space, next_obs))
        obs_hash = tuple(flatten(self.env.observation_space, obs))
        action_hash = tuple(flatten(self.env.action_space, action))

        # get the q value of the most likely action for the next observation state
        future_q_value = 0
        if not terminated and len(self.q_values[next_obs_hash]):
            next_obs_q_values = self.q_values[next_obs_hash]
            try:
                future_action_hash = max(next_obs_q_values, key=next_obs_q_values.get)
                future_q_value = self.q_values[next_obs_hash][future_action_hash]
            except:
                future_action_hash = max(next_obs_q_values, key=next_obs_q_values.get)
                pass

        # get the q value this observation state action
        q_value = 0
        if action_hash in self.q_values[obs_hash]:
            q_value = self.q_values[obs_hash][action_hash]

        # compute a delta
        temporal_difference = (
            reward + self.discount_factor * future_q_value - q_value
        )

        # update the q value for this observation state action
        self.q_values[obs_hash][action_hash] = (
            q_value + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
