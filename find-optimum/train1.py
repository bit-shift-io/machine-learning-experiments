import warnings
import os

warnings.filterwarnings('ignore')

from json import load
import pandas as pd
import numpy as np
import keras.backend as K
import tensorflow as tf
import math
import random
from keras import regularizers
from keras.layers import Input, Dense, Layer, Lambda
from keras.models import Model, load_model

import matplotlib.pyplot as plt


# This is just intended to generate some data for which the minima can be easily seen visually.
# It's a bit of a toy problem, but I think it's sufficient to illustrate the point. 
x = np.random.uniform(low=-5.0, high=5.0, size=(50,)) #np.arange(0, 1000)/1000.0 - 5
y = x ** 4 - 20 * x ** 2 +  10 * x + 4 + np.random.rand(len(x)) * 20
#plt.plot(x, y)


# reshape the matrix so it's compatible with keras
x = x.reshape(len(x), 1)


#import ipywidgets

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



     
# unsurprisingly it is able to replicate the original function
#plt.scatter(x, y)
#plt.scatter(x, p, color="red")


# we're going to freeze the model
model.trainable = False





# I don't really care about the input and output at this point 
sample_input = np.ones(10000).reshape(10000, 1)
#this doesn't matter at all, we're just going to minimize the function
sample_output = np.zeros(10000)



     


# this loss function is just the value of the function we're trying to minimize.
# all of the inputs are the same and so we don't really care about y_true
def my_loss_fn(y_true, y_pred):
    return tf.reduce_mean(y_pred, axis=-1) 



     



class custom_layer(Layer):
    def  __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(custom_layer,self).__init__(**kwargs)
    def build(self,input_shape):
        output_shape = self.compute_output_shape(input_shape)
        self.W=self.add_weight(name='kernel',
                           shape=(1,) + output_shape[1:],
                           initializer='uniform'                              ,
                           trainable=True)
        self.built = True
  # this self.built is necessary .
    def call(self,x):
        return x * self.W
    def compute_output_shape(self, input_shape):
        return(input_shape)



# input layer
new_input_layer =  Input(shape=(1,))
l = custom_layer(1)
# this is the layer whose weight we believe is going to be the minimum
new_weight_layer = l(new_input_layer)
# the model that we trained previously
transfer = model(new_weight_layer, training=False)

optimization_model = Model(new_input_layer, transfer)
optimization_model.compile('rmsprop', loss = my_loss_fn)
optimization_model.summary()
# basic training framework that is used in keras
optimization_model.fit(sample_input, sample_output, batch_size=4, epochs=5, validation_data=(x,y), verbose = 1)




optimum_x = l.get_weights()[0][0][0]
print(optimum_x)
optimum_y = model.predict(np.array([optimum_x]))



     


plt.scatter(x, y, s=2)
plt.scatter(x, p, color="red", s=2)

plt.axvline(x=optimum_x)
plt.axhline(y=optimum_y)



     
