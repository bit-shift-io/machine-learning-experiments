import warnings
import os

warnings.filterwarnings('ignore')

from json import load
import pandas as pd
import numpy as np
import keras.backend as K
import tensorflow as tf
import math

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Layer, Lambda
from tensorflow.keras.models import Model, load_model

import matplotlib.pyplot as plt


# This is just intended to generate some data for which the minima can be easily seen visually.
# It's a bit of a toy problem, but I think it's sufficient to illustrate the point. 
x = np.arange(0, 10000)/1000.0 - 5
y = x ** 4 - 20 * x ** 2 +  10 * x + 4 + np.random.rand(len(x)) * 20
plt.plot(x, y)


# reshape the matrix so it's compatible with keras
x = x.reshape(len(x), 1)


import ipywidgets

#this is our basic model that we're going to use to reconstruct the function

input_layer = Input(shape=(1,))
hidden_layer = Dense(12, activation='elu')(input_layer)
hidden_layer = Dense(9, activation='elu')(hidden_layer)
hidden_layer = Dense(6, activation='elu')(hidden_layer)
hidden_layer = Dense(3, activation='elu')(hidden_layer)
decoder = Dense(1, activation='linear')(hidden_layer)

model = Model(input_layer, decoder)
model.compile(loss='mse', optimizer='rmsprop')
model.fit(x, y, batch_size=16, epochs=20, validation_data=(x,y), verbose = 1)


model.fit(x, y, batch_size=4, epochs=20, validation_data=(x,y), verbose = 1)



# predict the curve again. Yes this is a self prediction, but we'll leave the cross validation as 
# an exercise for the reader
p = model.predict(x)



     
