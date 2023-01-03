import random
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten
import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque

class TrainAlgo_Base:
    """ Base class for training algorithm """

    def __init__(self, dnn, env):
        self.dnn = dnn
        self.env = env

    def train(self, n_episodes):
        pass

    def plot(self):
        pass



class TA_QL_Base(TrainAlgo_Base):
    """ Base class for qlearning training algorithm """

    def __init__(self, dnn, env, memory_maxlen=10000):
        self.dnn = dnn
        self.env = env

        self.gamma=.9
        self.epsilon=0.4
        self.eps_decay=0.995
        self.replay_size=100
        self.memory = deque(maxlen=memory_maxlen)

        # stats for plotting
        self.reward_total = []
        self.reward_avg = []
        self.reward_max = []
        self.reward_min = []
        self.rand_action_total = []
        self.action_total = []

    def train(self, n_episodes, fn_cb):
        t0 = time.time()

        for ei in range(n_episodes):
            terminated = self.train_episode(ei)
            if terminated:
                break

            self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
            fn_cb(ei)

        t1 = time.time()
        print(f"{n_episodes} episodes completed in {round((t1-t0)/60, 1)}min") 
            
            
    def train_episode(self, ei=0):
        """ Run an episode """
        t0 = time.time()
            
        self.train_episode_start()

        # Reset state
        state, info = self.env.reset()
        state = flatten(self.env.observation_space, state)

        result = False
        done = False
        reward_total = 0
        reward_max = None
        reward_min = None
        rand_action_total = 0
        action_total = 0
        
        while not done:
            action_total += 1

            # Implement greedy search policy to explore the state space
            if random.random() < self.epsilon:
                action = self.env.action_space.sample()
                #action = flatten(self.env.action_space, action)
                rand_action_total += 1
            else:
                q_values = self.dnn.predict([state])
                q_values_np = q_values.numpy()[0] #torch.argmax(q_values).item()
                splits = np.array_split(q_values_np, len(q_values_np) / self.env.n_actions)
                split_actions = list(map(lambda s: np.argmax(s), splits))

                action = {}
                for idx, lesson in enumerate(self.env.timetable.lessons):
                    id = f"lesson_{lesson.id}_actions"
                    action[id] = split_actions[idx]
            
            # Take action and add reward to total
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = flatten(self.env.observation_space, next_state)
            #next_state = np.reshape(next_state, [1, -1])
            
            # Update total and memory
            reward_total += reward
            if (reward_max == None or reward > reward_max):
                reward_max = reward
            if (reward_min == None or reward < reward_min):
                reward_min = reward

            flattened_action = flatten(self.env.action_space, action)
            self.memory.append((state, flattened_action, next_state, reward, terminated))
            q_values = self.dnn.predict([state]).tolist()
            
            # Update network weights using replay memory
            self.replay()

            if terminated:
                print("success, finish traning")
                result = True
                break

            if truncated:
                break

            state = next_state

        reward_avg = reward_total / action_total

        # append stats
        self.reward_total.append(reward_total)
        self.reward_avg.append(reward_avg)
        self.reward_max.append(reward_max)
        self.reward_min.append(reward_min)
        self.rand_action_total.append(rand_action_total)
        self.action_total.append(action_total)

        t1 = time.time()
        print(f"Ep {ei:<3}\t{round((t1-t0), 1)}s\tr.av: {round(reward_avg, 1):<5}\tr.mx: {reward_max:<5}\tr.mn: {reward_min:<5}") 
        return result

    def train_episode_start(self):
        pass

    
    def replay(self):
        pass


    def plot(self):
        ''' Plot the reward curve and histogram of results over time.'''
        plt.plot(self.reward_avg, label='reward avg', c='orange')
        plt.plot(self.reward_max, label='reward max', c='green')
        plt.plot(self.reward_min, label='reward min', c='blue')

        plt.axhline(self.env.max_hard_score, c='red',ls='--', label='goal')

        plt.axhline(0, c='black',ls='--', label='zero') # zero line
        plt.xlabel('Episodes')

        x = range(len(self.reward_avg))
        plt.legend()

        # Calculate the trend
        try:
            z = np.polyfit(x, self.reward_avg, 1)
            p = np.poly1d(z)
            plt.plot(x,p(x),"--", label='reward avg trend', c='orange')
        except:
            pass

        plt.show()
