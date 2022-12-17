import random
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten
import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import time

class TrainAlgo:
    """ Trains a DNN on an Env using Deep Q Learning """

    def __init__(self, dnn, env):
        self.dnn = dnn
        self.target = copy.deepcopy(self.dnn.model)

        self.env = env
        self.gamma=.9
        self.epsilon=0.3
        self.eps_decay=0.99
        self.replay_size=100
        self.memory = []

        # stats for plotting
        self.reward_total = []
        self.reward_avg = []
        self.reward_max = []
        self.reward_min = []
        self.rand_action_total = []
        self.action_total = []

    def train(self, n_episodes):
        for ei in range(n_episodes):
            self.train_episode(ei)
            self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
            
    def train_episode(self, ei):
        """ Run an episode """
        t0 = time.time()
            
        self.target_update()

        # Reset state
        state, info = self.env.reset()
        state = flatten(self.env.observation_space, state)

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
                rand_action_total += 1
            else:
                q_values = self.dnn.predict([state])
                action = torch.argmax(q_values).item()
            
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

            self.memory.append((state, action, next_state, reward, terminated))
            q_values = self.dnn.predict([state]).tolist()
            
            #t0=time.time()
            # Update network weights using replay memory
            self.replay()
            #t1=time.time()
            #sum_total_replay_time+=(t1-t0)

            if terminated:
                print("success, finish traning")
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
        print(f"Ep {ei} completed in {(t1-t0)}") 


    def target_update(self, TAU=0.3):
        ''' Update the targer gradually. '''
        updated_params = dict(self.target.named_parameters())
        for model_name, model_param in self.dnn.model.named_parameters():
            for target_name, target_param in self.target.named_parameters():
                if target_name == model_name:
                    # Update parameter
                    updated_params[model_name].data.copy_((TAU)*model_param.data + (1-TAU)*target_param.data)

        self.target.load_state_dict(updated_params)

    
    def replay(self):
        ''' Add experience replay to the DQL network class.'''
        memory = self.memory
        size = self.replay_size
        gamma = self.gamma
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.dnn.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            self.dnn.train(states, targets)


    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad():
            return self.target(torch.Tensor(s))


    def plot(self):
        ''' Plot the reward curve and histogram of results over time.'''
        plt.plot(self.reward_avg, label='reward avg', c='orange')

        plt.axhline(self.env.max_hard_score, c='red',ls='--', label='goal')

        plt.axhline(0, c='black',ls='--', label='zero') # zero line
        plt.xlabel('Episodes')
        #plt.ylabel('Actions to solution')

        #plt.plot(rand_actions_total, label="rand actions per run", c='black')
        #plt.plot(actions_total, label="total actions per run", c='green')

        x = range(len(self.reward_avg))
        plt.legend()
        # Calculate the trend
        #try:
        #    z = np.polyfit(x, actions_total, 1)
        #    p = np.poly1d(z)
        #    plt.plot(x,p(x),"--", label='actions trend', c='green')
        #except:
        #    print('')

        try:
            z = np.polyfit(x, self.reward_avg, 1)
            p = np.poly1d(z)
            plt.plot(x,p(x),"--", label='reward avg trend', c='orange')
        except:
            pass

        plt.show()


""""
def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=100, 
               title = 'DQL', double=False, 
               n_update=10, soft=False, verbose=True):
    "" "Deep Q Learning algorithm using the DQN. " " "
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
"""