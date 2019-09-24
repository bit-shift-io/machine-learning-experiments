import os, json, random
from PIL import Image
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
from keras.layers import Input, Embedding, LSTM, Dense
import keras.layers
import keras
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import operator
from enum import Enum
import json
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction import DictVectorizer

data_path = "./Data"


type_lb = LabelBinarizer()
type_lb.fit(["PrimaryResource", "IntermediateResource", "City"])

bool_lb = LabelBinarizer()
bool_lb.fit([True, False])


resource_count_max = 10


def add_tuple(a, b):
    return tuple(map(operator.add, a, b))


class Map:
    def __init__(self, map_path):
        with open(map_path, 'r') as f:
            self.json = json.load(f)

    def get_resource_node_as_vector(self, name):
        rn = next(rn for rn in self.json['resource_nodes'] if rn['name'] == name)
        type_t = type_lb.transform([rn['type']])
        resource = np.zeros(resource_count_max)
        i = 0
        for r in rn['resources']:
            is_demanded = bool_lb.transform([r['is_demanded']])[0][0]
            is_supplied = bool_lb.transform([r['is_supplied']])[0][0]
            resource[i + 0] = is_demanded
            resource[i + 1] = is_supplied
            i += 2
            nothing = 0


    def get_route_as_vector(self, route):
        nothing = 0


def load_data(set_dir_name):
    data = []
    set_path = os.path.join(data_path, set_dir_name)
    for o in os.listdir(set_path):
        map_path = os.path.join(set_path, o)
        if os.path.isfile(map_path):
            data.append(Map(map_path))

    return data


class KerasBatchGenerator(object):

    def __init__(self, data, batch_size):
        self.data = data
        self.current_data_idx = 0
        self.current_route_idx = 0

        self.batch_size = batch_size
        self.skip_step = 1
        self.num_steps = 1


    def generate(self):
        y = np.zeros((self.batch_size,))
        while True:
            for i in range(self.batch_size):
                if self.current_data_idx + self.num_steps >= len(self.data):
                    self.current_data_idx = 0

                current_data = self.data[self.current_data_idx]
                current_route = current_data.json['routes'][self.current_route_idx]

                rn_to_vec = current_data.get_resource_node_as_vector(current_route['to']['name'])
                rn_from_vec = current_data.get_resource_node_as_vector(current_route['from']['name'])
                x = np.concatenate(rn_to_vec, rn_from_vec)

                route_vec = current_data.get_route_as_vector(current_route)
                y = route_vec

                #x[i,] = self.data[self.current_idx:self.current_idx + self.num_steps]
                #temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                #y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)

                if self.current_route_idx + self.num_steps >= len(self.current_data.json['routes']):
                    self.current_route_idx = 0
                    self.current_data_idx += self.skip_step

            yield x, y



def plot_fit_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return


        
       
class Model1:
    def __init__(self, train_data, test_data):
        self.train_data = train_data 
        self.test_data = test_data 
       
        self.input_size = 100
        self.output_size = 100

        #to_categorical

        model = Sequential()
        model.add(Dense(self.input_size, input_shape=(self.input_size,)))
        model.add(PReLU())
        model.add(Dense(self.input_size))
        model.add(PReLU())
        model.add(Dense(self.output_size))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        #plot_model(model, to_file='Model1.png', show_shapes=True, show_layer_names=True)
        print(model.summary())
        self.model = model
        return


    def train(self):
        num_epochs = 50
        batch_size = 20
        num_samples = len(self.train_data)

        train_data_generator = KerasBatchGenerator(self.train_data, batch_size)
        valid_data_generator = KerasBatchGenerator(self.test_data, batch_size)

        checkpointer = ModelCheckpoint(filepath='Data/model-{epoch:02d}.hdf5', verbose=1)

        self.model.fit_generator(train_data_generator.generate(), math.ceil(num_samples / batch_size), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=math.ceil(num_samples / batch_size), callbacks=[checkpointer])

        self.model.save("Data/final_model.hdf5")

        plot_fit_history(history)
        return

    def load_weights(self):
        self.model.load_weights("Output/Model1.h5")

    def predict(self, map, history):
        input = np.array(map.map_1d)

        i = len(history) - 1
        for h in range(0, self.history_count):
            history_idx = i - h
            if (history_idx < 0):
                input = np.concatenate((input, np.zeros(map.map_1d.shape)))
            else:
                input = np.concatenate((input, history[history_idx]))

        # reshape inputs for the model
        input = input.reshape((-1, self.input_size))

        action_vec = self.model.predict(input)
        return get_action_from_vector(action_vec)