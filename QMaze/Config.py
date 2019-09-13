import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
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

# Small Q-test:
maze =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])


def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
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