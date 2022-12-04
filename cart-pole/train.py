# https://raw.githubusercontent.com/adventuresinML/adventures-in-ml-code/master/r_learning_python.py
# https://raw.githubusercontent.com/GaetanJUVIN/Deep_QLearning_CartPole/master/cartpole.py
# https://scientific-python.readthedocs.io/en/latest/notebooks_rst/6_Machine_Learning/04_Exercices/02_Practical_Work/02_RL_1_CartPole.html

import gym
import random
import time
import numpy as np
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

num_episodes = 100
for i in range(num_episodes):
    state, info = env.reset()
    env.render()

    done = False
    index = 0
    while not done:
        env.render()
        time.sleep(1/20)  # 20fps: Super slow for us poor little human!

        # make a prediction from keras model
        pred_state = np.reshape(state, [1, state_size])
        pred = model.predict(pred_state)
        action = np.argmax(pred)

        next_state, reward, done, truncated, info = env.step(action)

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
        # More innfo here: https://scientific-python.readthedocs.io/en/latest/notebooks_rst/6_Machine_Learning/04_Exercices/02_Practical_Work/02_RL_1_CartPole.html
        #
        
        state = next_state
        index += 1


print('Done')