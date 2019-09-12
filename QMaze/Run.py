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

pause_time = 0.5

qmaze = QMaze(maze)
rat_cell = random.choice(qmaze.free_cells)
qmaze.reset(rat_cell)

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