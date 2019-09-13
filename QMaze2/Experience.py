import numpy as np
from Config import *

class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]


    def get_inputs(self, mem_idx_head):
        envstate = self.memory[0][0]

        # push in the last 2 actions
        prev_action_0 = self.memory[len(self.memory) - (mem_idx_head + 1)][1]
        prev_action_1 = self.memory[len(self.memory) - (mem_idx_head + 2)][1]

        prev_0 = np.concatenate((envstate[0], [prev_action_0]))
        prev_1 = np.concatenate((envstate[0], [prev_action_1]))

        history = np.array([prev_1, prev_0])

        history = history.reshape(1, 2, envstate.size + 1)
        return history

    def predict(self, envstate, mem_idx_head=0):
        history = self.get_inputs(mem_idx_head)
        prediction = self.model.predict(history)
        return prediction[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, 2, env_size + 1)) # map size + 1 action
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = self.get_inputs(j)
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate, j)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
