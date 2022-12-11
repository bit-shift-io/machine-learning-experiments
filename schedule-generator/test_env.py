
from schedule_env import GridWorldEnv
import gym
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten

env3 = gym.envs.make("LunarLander-v2", render_mode="human")
env2 = gym.envs.make("FrozenLake-v1", render_mode="human")
env = gym.envs.make("gym_examples/GridWorld-v0", render_mode="human")

flat_space = flatten_space(env.observation_space)

state, info = env.reset()
state = flatten(env.observation_space, state)

done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = flatten(env.observation_space, next_state)

    if terminated or truncated:
        break