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
import tensorflow as tf
import keras.backend as K

print(tf.config.list_physical_devices())

# https://stackoverflow.com/questions/62927308/how-can-i-speed-up-my-model-training-process-using-tensorflow-and-keras
#tf.config.optimizer.set_jit(True)
#precision = 'float32'
#policy = tf.keras.mixed_precision.Policy(precision)
#tf.keras.mixed_precision.set_global_policy(policy)

env = gym.make('CartPole-v1', render_mode="human")

state_size        = env.observation_space.shape[0]
action_size       = env.action_space.n


# create the keras model
model = Sequential()
model.add(InputLayer(batch_input_shape=(1, state_size)))
#model.add(Dense(10, activation='sigmoid'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(Dense(2, activation='linear'))

# as time goes along, assume we get more skilled, so make smaller tweaks to our learning
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=0.00003,
#    decay_steps=1000, # roughly 10 steps per game with 100 games
#    decay_rate=0.9)
opt = tf.keras.optimizers.Adam(learning_rate=0.00001) #lr_schedule)#learning_rate=0.001)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy']) #metrics=['mae'])

# memory for each action - this tries to give us an even learning about each action
memory = [deque(maxlen=10000), deque(maxlen=10000)]

memory_size_before_post_episode_learn = 100
sample_batch_size = 32
num_episodes = 100
gamma = 0.95 # back propogate reward to previous states rate? deferred reward
exploration_rate = 1.0
exploration_min = 0.001
exploration_decay = 0.99
no_explore = False # every second episode see if we have arrived at a winning model

r_avg_list = []
r_best = 0
r_worst = 9999
r_learn_rate = []

with tf.device('/gpu:0'):
    for index_episode in range(num_episodes):
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])

        #env.render()
        
        done = False
        index = 0
        r_sum = 0

        while not done:
            #env.render()
            #time.sleep(1/30)  # 1/fps: Super slow for us poor little human!

            # decide if we should expore (go random) or predict from the model
            if np.random.rand() < exploration_rate:
                # future improvement search for the action we took in nthe past and use the opposite
                # if no opposite to choose form use our prediction
                action = random.randrange(action_size)
            else:
                # make a prediction from keras model
                #pred_state = np.reshape(state, [1, state_size])
                pred = model.predict(state, verbose=0)
                action = np.argmax(pred)

            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # learn from this step
            target = reward + gamma * np.max(model.predict(next_state, verbose=0))
            target_vec = model.predict(state, verbose=0)
            target_vec[0][action] = target
            model.fit(state, target_vec, epochs=1, verbose=0)

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
            # Deep Q-Learning seems to get stuck
            #
            # some onne here with a similar issue:
            #   https://ai.stackexchange.com/questions/34824/deep-q-learning-model-effectiveness-improves-then-crashes

            #if not no_explore:
            memory[action].append((state, action, reward, next_state, done))
            
            state = next_state
            index += 1
            r_sum += reward


        #if r_sum > r_best:
        #    r_best = r_sum
        #    # if we are improving, reduce how much we learn
        #    K.set_value(model.optimizer.learning_rate, model.optimizer.learning_rate * 0.5)
        #    print("We are improving...")
        #if r_sum < r_worst:
        #    r_worst = r_worst

        #else if r_sum < r_best:
        #    K.set_value(model.optimizer.learning_rate, model.optimizer.learning_rate * 0.5)

        r_avg_list.append(r_sum / 1000)

        learn_rate = K.eval(model.optimizer.lr) #K.eval(model.optimizer._decayed_lr(tf.float64))
        #r_learn_rate.append(learn_rate)
        print("Episode {}# Score: {} Reward: {} Learn rate: {} Explore rate: {}".format(index_episode, index + 1, r_sum, learn_rate, exploration_rate))

        # now learn from the memory
        for a_idx in range(action_size):
            if len(memory[a_idx]) >= memory_size_before_post_episode_learn:
                print(f"Post Episode Learning...Action: {a_idx}")

                # make sure we learn some from every action
                sample_batch = random.sample(memory[a_idx], sample_batch_size)
                for state, action, reward, next_state, done in sample_batch:
                    target = reward
                    if not done:
                        pred = model.predict(next_state, verbose=0)
                        target = reward + gamma * np.amax(pred[0])
                    target_f = model.predict(state, verbose=0)
                    target_f[0][action] = target
                    model.fit(state, target_f, epochs=1, verbose=0)

        # modify exportation - we explore less as we get more skilled
        if exploration_rate > exploration_min:
            exploration_rate *= exploration_decay

    plt.plot(r_avg_list)
    plt.ylabel('Average reward per game')
    plt.xlabel('Number of games')
    plt.show()

print('Done')