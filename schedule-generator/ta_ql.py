import random
from gym.spaces.utils import flatdim, flatten, flatten_space, unflatten
import torch
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
from train_algo import TA_QL_Base

class TA_QL(TA_QL_Base):
    """ deep q learning """

    def __init__(self, dnn, env):
        super().__init__(dnn, env)
 
    def replay(self):
        memory = self.memory
        size = self.replay_size
        gamma = self.gamma

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
        
            all_q_values = self.dnn.predict(states) # predicted q_values of all states
            all_q_values_next = self.dnn.predict(next_states) # predict next state values

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
            
            self.dnn.train(states.tolist(), all_q_values.tolist())


