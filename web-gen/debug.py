import matplotlib.pyplot as plt
from utils import *
from config import *
import math


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch 
import torch.nn as nn
import torchvision
#import datasets in torchvision
import torchvision.datasets as datasets

#import model zoo in torchvision
import torchvision.models as models
import torchvision.transforms as transforms

def to_bounds(size):
    return [0, 0, size[0], size[1]]

# https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
def create_corner_rect(bounds, color='red'):
    xy = from_fractional_scale(bounds, image_size)
    rect = plt.Rectangle((xy[0], xy[1]), xy[2], xy[3], edgecolor=color, facecolor='none', lw=1)
    return rect


def create_subplots(size):
    l = size
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
    


def show_data(image, actual_first_child_size=None, pred_first_child_size=None, block=False, pause=0.6):
    plt.clf()
    plt.imshow(image.permute(1, 2, 0))

    if actual_first_child_size is not None:
        plt.gca().add_patch(create_corner_rect(to_bounds(actual_first_child_size)))

    if pred_first_child_size is not None:
        plt.gca().add_patch(create_corner_rect(to_bounds(pred_first_child_size), 'green'))

    plt.show(block=block)
    plt.pause(pause)




## CONV2d
# https://github.com/Niranjankumar-c/DeepLearning-PadhAI/blob/master/DeepLearning_Materials/6_VisualizationCNN_Pytorch/CNNVisualisation.ipynb

def plot_filters_single_channel_big(t):
    
    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]
    
    
    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)
    
    npimg = npimg.T
    
    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)


def plot_filters_single_channel(t):
    
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    fig = plt.figure(figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
   
    plt.tight_layout()
    plt.show()

def plot_filters_multi_channel(t):
    
    #get the number of kernals
    num_kernels = t.shape[0]    
    
    #define number of columns for subplots
    num_cols = 12
    #rows = num of kernels
    num_rows = num_kernels
    
    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
        #for each kernel, we convert the tensor to numpy 
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
    plt.savefig('myimage.png', dpi=100)    
    plt.tight_layout()
    plt.show()

def plot_weights(model, layer_num, single_channel = True, collated = False):
  
  #extracting the model features at the particular layer number
  layer = model[layer_num]
  
  #checking whether the layer is convolution layer or not 
  if isinstance(layer, nn.Conv2d):
    #getting the weight tensor data
    weight_tensor = model[layer_num].weight.data
    
    if single_channel:
      if collated:
        plot_filters_single_channel_big(weight_tensor)
      else:
        plot_filters_single_channel(weight_tensor)
        
    else:
      if weight_tensor.shape[1] == 3:
        plot_filters_multi_channel(weight_tensor)
      else:
        print("Can only plot weights with three channels with single channel = False")
        
  else:
    print("Can only visualize layers which are convolutional")