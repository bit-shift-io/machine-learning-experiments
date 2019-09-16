import os
from PIL import Image
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers import Input, Embedding, LSTM, Dense
import keras.layers
import keras
import matplotlib.pyplot as plt

data_path = "./Data"
map_size = (7, 7)

# https://stackoverflow.com/questions/15612373/convert-image-png-to-matrix-and-then-to-1d-array
def read_img(file_path):
    img = Image.open(file_path).convert(mode='L').convert(mode='F')
    np_im = np.array(img)
    np_im /= 255.0
    nothing = 0
    return np_im


class Path:
    def __init__(self, path):
        self.path = path
        self.image = read_img(path)
        return


class Map:
    def __init__(self, path):
        self.path = path
        self.map = read_img(os.path.join(path, "Map.png"))

        self.paths = []
        for o in os.listdir(path):
            path_path = os.path.join(path, o)
            if os.path.isfile(path_path) and o.startswith("Path"):
                self.paths.append(Path(path_path))

        return


def load_data():
    maps = []
    for o in os.listdir(data_path):
        map_path = os.path.join(data_path, o)
        if os.path.isdir(map_path):
            maps.append(Map(map_path))

    return maps


class PFModel:
    def __init__(self):
        self.model = self.build_model()


    def build_model(self):
        # https://keras.io/getting-started/functional-api-guide/

        # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
        # Note that we can name any layer by passing it a "name" argument.
        main_input = Input(shape=(100,), dtype='int32', name='main_input')

        # This embedding layer will encode the input sequence
        # into a sequence of dense 512-dimensional vectors.
        x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

        # A LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        lstm_out = LSTM(32)(x)

        auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

        auxiliary_input = Input(shape=(5,), name='aux_input')
        x = keras.layers.concatenate([lstm_out, auxiliary_input])

        # We stack a deep densely-connected network on top
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)

        model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

        model.compile(optimizer='rmsprop',
                loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
                loss_weights={'main_output': 1., 'aux_output': 0.2})

        print(model.summary())
        return model


    def train(self):
        # And trained it via:
        self.model.fit({'main_input': headline_data, 'aux_input': additional_data},
            {'main_output': labels, 'aux_output': labels},
            epochs=50, batch_size=32)
