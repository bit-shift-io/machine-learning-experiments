import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import glob
import json
from PIL import Image
import torchvision.transforms.functional as FT
from torchvision import transforms
import torch
import torch.nn.functional as F
from utils import *
from pathlib import Path
from sklearn import preprocessing
from config import *


# keys = str, list, tuple of one or more keys
# voc = vocabulary = list all the available classes
def one_hot(voc, keys):
    if isinstance(keys, str):
        keys = [keys]

    le = preprocessing.LabelEncoder()
    le.fit(voc)

    xform_keys = le.transform(keys)
    oh = F.one_hot(torch.tensor(xform_keys, dtype=torch.int64), num_classes=len(voc))
    return oh.type(torch.float32)

def one_hot_inverse(voc, oh):
    le = preprocessing.LabelEncoder()
    le.fit(voc)
    keys = le.inverse_transform(oh)
    return keys


class Transformer:
    def __init__(self, image_size):
        self.image_size = image_size
        return


    # encode inputs ready for NN to consume
    def encode_input_image(self, image):
        image = FT.resize(image, self.image_size)
        image = transforms.ToTensor()(image) # convert from 0->255 to 0->1
        return image

    # image already resized by crawler
    def encode_input_image_200(self, image):
        image = transforms.ToTensor()(image) # convert from 0->255 to 0->1
        return image


    def encode_bounds(self, parent_wh, bounds_arr):
        frac_bounds_arr = to_fractional_scale(bounds_arr, parent_wh)
        centre_bounds_arr = xy_to_cxcy(frac_bounds_arr)
        bounds = torch.FloatTensor(centre_bounds_arr)
        return bounds

    def encode_node_class(self, node_cls):
        return one_hot(node_classes, node_cls)[0]

    def decode_node_class(self, node_one_hot):
        node_cls = one_hot_inverse(node_classes, [node_one_hot.argmax()])[0]
        return node_cls

    def encode_display_class(self, display_cls):
        return one_hot(display_classes, display_cls)[0]


    def encode_layout_class(self, layout):
        return one_hot(layout_classes, layout)[0]

    def decode_layout_class(self, layout_oh):
        r = one_hot_inverse(display_classes, [layout_oh.argmax()])[0]
        return r

    def encode_first_child_size(self, first_child_size):
        return torch.FloatTensor(first_child_size)

    # # encode expected NN outputs for training
    # def encode_outputs(self, parent_wh, bounds_arr, node_cls, display_cls):
    #     frac_bounds_arr = to_fractional_scale(bounds_arr, parent_wh)
    #     centre_bounds_arr = xy_to_cxcy(frac_bounds_arr)
    #     bounds = torch.FloatTensor(centre_bounds_arr)

    #     node_onehot = one_hot(node_classes, node_cls)[0]
    #     display_onehot = one_hot(display_classes, display_cls)[0]

    #     labels = torch.cat((bounds, node_onehot, display_onehot), dim=0)
    #     return labels


    # convert NN output to a human understandable output
    def decode_output(self, Y):
        r = []
        for i, y in enumerate(Y):
            bounds = y[0:4]
            node_onehot = one_hot_inverse(node_classes, [y[4:6].argmax()])[0]
            display_onehot = one_hot_inverse(display_classes, [y[6:8].argmax()])[0]
            r.append([bounds, node_onehot, display_onehot])

        return r

    # in features for NN
    def input_size(self):
        return self.image_size

    # the number of outputs the model needs for the labels (out features for NN)
    def output_size(self):
        return bounds_len + len(node_classes) + len(display_classes)