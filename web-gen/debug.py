import matplotlib.pyplot as plt
from utils import *
from config import *
import math

def to_bounds(size):
    return [0, 0, size[0], size[1]]

# https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
def create_corner_rect(bounds, color='red'):
    xy = from_fractional_scale(bounds, image_size)
    rect = plt.Rectangle((xy[0], xy[1]), xy[2], xy[3], edgecolor=color, facecolor='none', lw=1)
    return rect


def create_subplots(image_arr):
    l = len(image_arr)
    s = math.ceil(math.sqrt(l))
    f, axarr = plt.subplots(s, s)
    
    return axarr


def show_data_grid(axarr, image_arr, actual_first_child_size_arr, pred_first_child_size_arr=None):
    l = len(image_arr)
    s = math.ceil(math.sqrt(l))
    ss = s * s

    for i in range(0, l):
        x = i % s
        y = math.floor(i / s)
        subplot = axarr[y, x]
        subplot.set_visible(True)
        subplot.cla()

        image = image_arr[i]
        subplot.imshow(image.permute(1, 2, 0))

        # https://stackoverflow.com/questions/62991535/how-to-draw-a-rectangular-on-subplotted-figure-using-matlibplot-in-python
        actual_first_child_size = actual_first_child_size_arr[i]
        subplot.add_patch(create_corner_rect(to_bounds(actual_first_child_size)))
        
        if pred_first_child_size_arr is not None:
            pred_first_child_size = pred_first_child_size_arr[i]
            subplot.add_patch(create_corner_rect(to_bounds(pred_first_child_size), 'green'))

    # hide unused subplots
    for i in range(l, ss):
        x = i % s
        y = math.floor(i / s)
        subplot = axarr[y, x]
        subplot.set_visible(False)

    plt.show(block=False)
    plt.pause(0.1)
    


def show_data(image, actual_first_child_size, pred_first_child_size=None):
    plt.clf()
    plt.imshow(image.permute(1, 2, 0))
    plt.gca().add_patch(create_corner_rect(to_bounds(actual_first_child_size)))

    if pred_first_child_size is not None:
        plt.gca().add_patch(create_corner_rect(to_bounds(pred_first_child_size), 'green'))

    plt.show(block=False)
    plt.pause(0.6)