# https://raw.githubusercontent.com/adventuresinML/adventures-in-ml-code/master/r_learning_python.py
# https://raw.githubusercontent.com/GaetanJUVIN/Deep_QLearning_CartPole/master/cartpole.py
# https://scientific-python.readthedocs.io/en/latest/notebooks_rst/6_Machine_Learning/04_Exercices/02_Practical_Work/02_RL_1_CartPole.html

import gym
import random
import time
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import matplotlib.pylab as plt

env = gym.make('CartPole-v1', render_mode="human")

state_size        = env.observation_space.shape[0]
action_size       = env.action_space.n


# create the keras model
model = Sequential()
model.add(InputLayer(batch_input_shape=(1, state_size)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

memory = deque(maxlen=2000)
sample_batch_size = 32
num_episodes = 100
gamma = 0.95 # back propogate reward to previous states rate? deferred reward
exploration_rate = 1.0
exploration_min = 0.01
exploration_decay = 0.99

for i in range(num_episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])

    env.render()

    done = False
    index = 0
    while not done:
        env.render()
        time.sleep(1/30)  # 1/fps: Super slow for us poor little human!

        # decide if we should expore (go random) or predict from the model
        if np.random.rand() <= exploration_rate:
            action = random.randrange(action_size)
        else:
            # make a prediction from keras model
            #pred_state = np.reshape(state, [1, state_size])
            pred = model.predict(state)
            action = np.argmax(pred)

        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # train?
        #
        # Deep Q-Learning....
        # here: https://raw.githubusercontent.com/GaetanJUVIN/Deep_QLearning_CartPole/master/cartpole.py
        #       https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
        #   traning is completed after a whole play/run of the episode
        #
        # Q-Learning....
        # here: https://raw.githubusercontent.com/adventuresinML/adventures-in-ml-code/master/r_learning_python.py
        #   training is done on the fly
        #
        # More info here: https://scientific-python.readthedocs.io/en/latest/notebooks_rst/6_Machine_Learning/04_Exercices/02_Practical_Work/02_RL_1_CartPole.html
        #
        # Deep Q-Learning is the more modern approach and requires less knowledge as we store stuff into memory

        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        index += 1

    # now learn from the memory
    if len(memory) >= sample_batch_size:
        sample_batch = random.sample(memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + gamma * np.amax(model.predict(next_state)[0])
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

        # modify exportation - we explore less as we get more skilled
        if exploration_rate > exploration_min:
            exploration_rate *= exploration_decay


print('Done')