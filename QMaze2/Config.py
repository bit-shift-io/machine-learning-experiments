import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Embedding, Dropout
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from PIL import Image

rat_mark = 0.5      # The current rat cell will be painteg by gray 0.5

visited_mark = 0.8  # Cells visited by the rat will be painted by gray 0.8
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1

def build_model(maze, lr=0.001):

    embed_dim = 128
    lstm_out = 200
    batch_size = 32

    # 1 feature, with 10 timesteps
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    data = data.reshape((1, 10, 1))

    # 2 features, with 10 time steps
    data2 = np.array([
      [0.1, 1.0],
      [0.2, 0.9],
      [0.3, 0.8],
      [0.4, 0.7],
      [0.5, 0.6],
      [0.6, 0.5],
      [0.7, 0.4],
      [0.8, 0.3],
      [0.9, 0.2],
      [1.0, 0.1]])
    data2 = data2.reshape(1, 10, 2)
    print(data2.shape)

    model = Sequential()

    # https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
    # the model below defines an input layer that expects 1 or more samples, 2 time steps, and (map.size + 1 action) features (map + action). 
    model.add(LSTM((maze.size + 1) * 2, input_shape=(2, maze.size + 1)))
    model.add(Dropout(0.2)) # apparently helps reduce overfit

    

    #model.add(Embedding(2500, embed_dim,input_length = maze.size, dropout = 0.2))
    #model.add(Dense(maze.size, input_shape=(maze.size,)))
    #model.add(LSTM(units = 1, return_sequences = True, input_shape = (maze.size, 1)))
    
    #LSTM(maze.size, input_shape=(None, maze.size, 1), return_sequences=True))
    #model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
    #model.add(Dense(num_actions, activation='softmax'))
    #model.compile(loss = 'mse', optimizer='adam',metrics = ['accuracy'])

    #model = Sequential()
    #model.add(LSTM(maze.size, input_shape=(maze.size,), return_sequences=True))
    #model.add(LSTM(maze.size), return_sequences=True)
    #model.add(Dense(num_actions))

    #model.add(Dense(maze.size, input_shape=(maze.size,)))
    #model.add(PReLU())
    #model.add(Dense(maze.size))
    #model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model


def show(qmaze, pause_time = 0.001):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3   # rat cell
    canvas[nrows-1, ncols-1] = 0.9 # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')

    #plt.ion() # https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
    plt.show(block = False)
    plt.draw()
    plt.pause(pause_time)
    return img


# https://stackoverflow.com/questions/15612373/convert-image-png-to-matrix-and-then-to-1d-array
def read_img(file_path):
    img = Image.open(file_path).convert(mode='L').convert(mode='F')
    np_im = np.array(img)
    np_im /= 255.0
    nothing = 0
    return np_im
    #img = plt.imread(file_path)
  #  rows,cols,colors = img.shape # gives dimensions for RGB array
  #  img_size = rows*cols*colors
  #  img_1D_vector = img.reshape(img_size)
    # you can recover the orginal image with:
  #  img2 = img_1D_vector.reshape(rows,cols,colors)

   # plt.imshow(img) # followed by 
   # plt.show() # to show the first image, then 
   # plt.imshow(img2) # followed by
   # plt.show() # to show you the second image.