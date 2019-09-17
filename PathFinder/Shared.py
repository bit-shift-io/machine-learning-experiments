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
import operator
from enum import Enum


data_path = "./Data"
map_size = (7, 7)


# https://stackoverflow.com/questions/15612373/convert-image-png-to-matrix-and-then-to-1d-array
def read_img(file_path):
    img = Image.open(file_path).convert(mode='L').convert(mode='F')
    np_im = np.array(img)
    np_im /= 255.0
    nothing = 0
    return np_im



class PathAction(Enum):
    NONE = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    MAX = 5

action_direction_map = {
    PathAction.NONE: (0, 0),
    PathAction.UP: (0, -1),
    PathAction.RIGHT: (1, 0),
    PathAction.DOWN: (0, 1),
    PathAction.LEFT: (-1, 0)
}


class Path:
    def __init__(self, path):
        self.path = path
        self.image = read_img(path)
        self.compiled_path = self.compile_from_image()
        return

    def compile_from_image(self):
        pos = (0, 0)
        
        location_history = []
        action_history = []

        image_shape = self.image.shape     
        end_pos = tuple(map(operator.add, image_shape, (-1, -1)))

        while (pos != end_pos):
            for a in range(PathAction.UP.value, PathAction.LEFT.value):
                action = PathAction(a)
                direction = action_direction_map[action]

                next_pos = tuple(map(operator.add, pos, direction))

                # do not go backwards!
                if (len(location_history) and location_history[-1] == next_pos):
                    continue

                # out of bounds check
                if (next_pos[0] < 0 or next_pos[0] >= image_shape[0]
                    or next_pos[1] < 0 or next_pos[1] >= image_shape[1]):
                    continue

                # we found the next pixel we haven't been too yet
                pixel = self.image[next_pos]
                if (pixel < 0.5):
                    location_history.append(pos)
                    action_history.append(action)
                    pos = next_pos
                    break

        self.location_history = location_history
        self.action_history = action_history
        return

    def get_action_history_values(self):
        values = list(map(lambda a: a.value, self.action_history))
        return values



class Map:
    def __init__(self, path):
        self.path = path
        self.map_2d = read_img(os.path.join(path, "Map.png"))
        self.map_1d = self.map_2d.reshape((-1,)) # convert from 2d to 1d array

        self.paths = []
        for o in os.listdir(path):
            path_path = os.path.join(path, o)
            if os.path.isfile(path_path) and o.startswith("Path"):
                self.paths.append(Path(path_path))

        return


class DataSet:
    def __init__(self):
        self.maps = self.load_data()
        return


    def load_data(self):
        maps = []
        for o in os.listdir(data_path):
            map_path = os.path.join(data_path, o)
            if os.path.isdir(map_path):
                maps.append(Map(map_path))

        return maps


    def get_const_input_shape(self):
        first_map = self.maps[0]
        shape = first_map.map_1d.shape
        return shape




        
class PFModel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = self.build_model()        

    def build_model(self):
        # https://keras.io/getting-started/functional-api-guide/

        # Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
        # Note that we can name any layer by passing it a "name" argument.
        temporal_input = Input(shape=(100,), dtype='int32', name='temporal_input')

        # This embedding layer will encode the input sequence
        # into a sequence of dense 512-dimensional vectors.
        x = Embedding(output_dim=512, input_dim=10000, input_length=100)(temporal_input)

        # A LSTM will transform the vector sequence into a single vector,
        # containing information about the entire sequence
        lstm_out = LSTM(32, input_dim=2)(x)

        #temporal_output = Dense(1, activation='sigmoid', name='temporal_output')(lstm_out) # FMNOTE: commented out

        const_input_shape = self.dataset.get_const_input_shape()
        const_input = Input(shape=const_input_shape, name='const_input')
        x = keras.layers.concatenate([lstm_out, const_input])

        # We stack a deep densely-connected network on top
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)

        model = Model(inputs=[temporal_input, const_input], outputs=[main_output]) # , temporal_output # FMNOTE: commented out

        model.compile(optimizer='rmsprop',
                loss={'main_output': 'binary_crossentropy'}, # , 'temporal_output': 'binary_crossentropy' # FMNOTE: commented out
                loss_weights={'main_output': 1.}) # , 'temporal_output': 0.2 # FMNOTE: commented out

        print(model.summary())
        return model


    def train(self):
        # compile data into appropriate lists
        temporal_inputs = []
        const_inputs = []
        main_outputs = []
        temporal_outputs = []
        for map in self.dataset.maps:
            for path in map.paths:
                const_inputs.append(map.map_1d)
                temporal_inputs.append(path.location_history)
                main_outputs.append(path.get_action_history_values())

        # And trained it via:
        self.model.fit({'temporal_input': temporal_inputs, 'const_input': const_inputs},
            {'main_output': main_outputs}, # , 'temporal_output': temporal_outputs # FMNOTE: commented out
            epochs=50, batch_size=32)
