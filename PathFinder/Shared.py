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
        self.map = read_img(os.path.join(path, "Map.png"))

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


class History:
    def __init__(self):
        self.history = []
        return

        
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


    def train(self, dataset):
        # compile data into appropriate lists
        aux_inputs = []
        aux_outputs = []
        for map in dataset.maps:
            for path in map.paths:
                aux_inputs.append(map.map)
                aux_outputs.append(path.get_action_history_values())


        # And trained it via:
        self.model.fit({'main_input': headline_data, 'aux_input': additional_data},
            {'main_output': labels, 'aux_output': labels},
            epochs=50, batch_size=32)
