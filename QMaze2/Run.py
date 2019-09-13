#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Vizusalise the model 

import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt

# lets get the trained model and make an animated gif
# https://www.mathworks.com/matlabcentral/answers/94495-how-can-i-create-animated-gif-images-in-matlab

from Config import *
from QMaze import *
from Experience import *

pause_time = 0.3

maze = read_img("Data/Maze_3.png") #random.choice(["Data/Maze_2.png", "Data/Maze_1.png"]))
qmaze = QMaze(maze)

model = build_model(maze)
model.load_weights("model.h5")

show(qmaze, pause_time)

envstate = qmaze.observe()
while True:
    prev_envstate = envstate
    # get next action
    q = model.predict(prev_envstate)
    action = np.argmax(q[0])

    # apply action, get rewards and new state
    envstate, reward, game_status = qmaze.act(action)

    show(qmaze, pause_time)

    if game_status == 'win':
        print(game_status)
        break
    elif game_status == 'lose':
        print(game_status)
        break