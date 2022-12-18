import random
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten
import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
from train_algo import TA_QL_Base

class TA_QL_DoubleSoft(TA_QL_Base):
    """ Double soft deep q learning """

    def __init__(self, dnn, env, TAU=0.3):
        super().__init__(dnn, env)
        self.target = copy.deepcopy(self.dnn.model)
        self.TAU = TAU
 

    def train_episode_start(self):
        ''' Update the targer gradually. '''
        updated_params = dict(self.target.named_parameters())
        for model_name, model_param in self.dnn.model.named_parameters():
            for target_name, target_param in self.target.named_parameters():
                if target_name == model_name:
                    # Update parameter
                    updated_params[model_name].data.copy_((self.TAU)*model_param.data + (1-self.TAU)*target_param.data)

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

