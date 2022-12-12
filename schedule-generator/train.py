import gym
import copy
import torch
from torch.autograd import Variable
import random

# https://stackoverflow.com/questions/7534453/matplotlib-does-not-show-my-plot-although-i-call-pyplot-show
# possibly fixing plt randomly not showing?
import matplotlib
matplotlib.use("MacOSX")

import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import time

from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten

from timetable_env import TimeTableEnv
from schedule_env import GridWorldEnv

#env = gym.envs.make("LunarLander-v2", render_mode="human")
#env = gym.envs.make("FrozenLake-v1", render_mode="human")
#env = gym.envs.make("gym_examples/GridWorld-v0", render_mode="human")
env = gym.envs.make("TimeTable-v0", render_mode="human")

def plot_res(values, title='', actions_total=[], rand_actions_total=[]):   
    ''' Plot the reward curve and histogram of results over time.'''
    plt.plot(values, label='score per run', c='orange')
    plt.axhline(env.max_hard_score, c='red',ls='--', label='goal')
    plt.axhline(0, c='black',ls='--', label='goal') # zero line
    plt.xlabel('Episodes')
    plt.ylabel('Reward')

    plt.plot(rand_actions_total, label="rand actions per run", c='black')
    plt.plot(actions_total, label="total actions per run", c='green')

    x = range(len(values))
    plt.legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"--", label='trend', c='orange')
    except:
        print('')

    plt.show()


class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=0.05):
            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim*2),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim*2, action_dim)
                    )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def replay(self, memory, size, gamma=0.9):
        """New replay function"""
        #Try to improve replay speed
        if len(memory) >= size:
            batch = random.sample(memory, size)
            batch_t = list(map(list, zip(*batch))) #Transpose batch list
            states = batch_t[0]
            actions = batch_t[1]
            next_states = batch_t[2]
            rewards = batch_t[3]
            is_dones = batch_t[4]
        
            states = torch.Tensor(states)
            actions_tensor = torch.Tensor(actions)
            next_states = torch.Tensor(next_states)
            rewards = torch.Tensor(rewards)
            is_dones_tensor = torch.Tensor(is_dones)
        
            is_dones_indices = torch.where(is_dones_tensor==True)[0]
        
            all_q_values = self.model(states) # predicted q_values of all states
            all_q_values_next = self.model(next_states) # predict next state values

            # Update q values
            #
            # this applies the forumula:
            #   [desired action value] = reward + gamma * [highest predicted action value]
            # to each sample in the batch to create a traning set of data for the model to learn
            #
            value_of_highest_next_action = torch.max(all_q_values_next, axis=1).values
            q_v = rewards + gamma * value_of_highest_next_action
            q_r = rewards[is_dones_indices.tolist()]

            all_q_values[range(len(all_q_values)), actions] = q_v
            all_q_values[is_dones_indices.tolist(), actions_tensor[is_dones].tolist()] = q_r
            
            self.update(states.tolist(), all_q_values.tolist())

class DQN_double(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        super().__init__(state_dim, action_dim, hidden_dim, lr)
        self.target = copy.deepcopy(self.model)
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad():
            return self.target(torch.Tensor(s))
        
    def target_update(self):
        ''' Update target network with the model weights.'''
        self.target.load_state_dict(self.model.state_dict())
        
    def replay(self, memory, size, gamma=1.0):
        ''' Add experience replay to the DQL network class.'''
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            self.update(states, targets)


class DQN_double_soft(DQN_double):
    def target_update(self, TAU=0.3):
        ''' Update the targer gradually. '''
        updated_params = dict(self.target.named_parameters())
        for model_name, model_param in self.model.named_parameters():
            for target_name, target_param in self.target.named_parameters():
                if target_name == model_name:
                    # Update parameter
                    updated_params[model_name].data.copy_((TAU)*model_param.data + (1-TAU)*target_param.data)

        self.target.load_state_dict(updated_params)




def perform_episode(env, model, epsilon, memory, replay, replay_size, gamma):
    """ Run an episode """

    # Reset state
    state, info = env.reset()
    state = flatten(env.observation_space, state)

    #memory = []
    done = False
    total = 0
    rand_action_total = 0
    action_total = 0
    
    while not done:
        action_total += 1

        # Implement greedy search policy to explore the state space
        if random.random() < epsilon:
            action = env.action_space.sample()
            rand_action_total += 1
        else:
            q_values = model.predict([state])
            action = torch.argmax(q_values).item()
        
        # Take action and add reward to total
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = flatten(env.observation_space, next_state)
        #next_state = np.reshape(next_state, [1, -1])
        
        # Update total and memory
        total += reward
        memory.append((state, action, next_state, reward, terminated))
        q_values = model.predict([state]).tolist()
            
        if terminated or truncated:
            break

        t0=time.time()
        # Update network weights using replay memory
        model.replay(memory, replay_size, gamma)
        t1=time.time()
        #sum_total_replay_time+=(t1-t0)

        state = next_state

    return memory, total, rand_action_total, action_total


def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=100, 
               title = 'DQL', double=False, 
               n_update=10, soft=False, verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    rand_actions_total = []
    actions_total = []
    memory = []
    episode_i=0
    sum_total_replay_time=0
    for episode in range(episodes):
        episode_i+=1
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        if double and soft:
            model.target_update()
        
        memory_at_replay_size = len(memory) >= replay_size

        ep_memory, total, rand_action_total, action_total = perform_episode(env, model, epsilon if memory_at_replay_size else 1.0, memory, replay, replay_size, gamma)
       
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        rand_actions_total.append(rand_action_total)
        actions_total.append(action_total)

        if verbose:
            print(f"episode: {episode}, total reward: {total}, actions: {action_total}")


    plot_res(final, title, actions_total, rand_actions_total)
    
    if verbose:
        print("episode: {}, total reward: {}".format(episode_i, total))
        if replay:
            print("Average replay time:", sum_total_replay_time/episode_i)
        
    return final


# Number of states
n_state = flatten_space(env.observation_space).shape[0]
# Number of actions
n_action = env.action_space.n
# Number of episodes
episodes = 100
# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001



# Get DQN results
simple_dqn = DQN_double_soft(n_state, n_action, n_hidden, lr)
simple = q_learning(env, simple_dqn, episodes, gamma=.9, epsilon=0.3, replay_size=100, replay=True, double=True, soft=True, 
    title='Double DQL with Replay', verbose=True)



     
