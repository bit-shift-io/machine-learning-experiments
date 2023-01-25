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
import random

# randomly shufle RGB channels
# https://discuss.pytorch.org/t/torch-tensor-variable-from-rgb-to-bgr/18955
class RandomColourShuffle(object):
    def __call__(self, sample):
        permute = [2, 1, 0]
        random.shuffle(permute)
        modified = sample[permute, :]
        return modified

# sample: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py
class WebsitesDataset(Dataset):
    def __init__(self, data_dir, transformer, layouts_only=True):
        search = f'{data_dir}/**/*.json'
        # for now, lets ignore leaf nodes and containers as it is most important to learn layouts
        if layouts_only:
            search = f'{data_dir}/**/layout.json'

        self.samples = glob.glob(search, recursive = True)
        self.transformer = transformer
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with open(sample, 'r') as f:
            js = json.load(f)


        # compute the first child ouputs
        # i.e. when we see see our screenshot, what would we expect as the output?
        # that should be:
        #   1) a bounding region for the children
        #   2) any css properties for this element (ignore for now)
        #   3) the tag name for this elemennt (ignnore for now)
        #
        # then we can take a sub-image using the bounding region of the children and repeat until
        # no results are returned
        #
        # how do we deal with multiple children?
        # we need to know if it is a display: flex, grid or block layout

        # parent_wh = [js['parent_size']['width'], js['parent_size']['height']] #[parent_js['bounds']['width'], parent_js['bounds']['height']]

        # # for now return bounds of first child
        # try:
        #     children = js['children'] if 'children' in js else []
        #     first_child = children[0]
        #     bounds_arr = [first_child['bounds']['x'], first_child['bounds']['y'], first_child['bounds']['width'], first_child['bounds']['height']]
        #     parent_wh = [first_child['parent_size']['width'], first_child['parent_size']['height']]
        # except:
        #     # if not children we end up here.... return bounds that encompass the whole image
        #     bounds_arr = [0, 0, parent_wh[0], parent_wh[1]]
        

        # TODO:
        # dynamic modify data on nthe fly
        # swap rgba channels
        # convert to greyscale
        # scale/skew

        # the sample code above applies random variation and flips etc...
        # do we need to do something similar to help AI in fuzzy situations?
        image = Image.open(js['img_path_200'])
        X = self.transformer.encode_input_image_200(image)

        # img_action = random.randint(0, 1)
        # if img_action == 0: # convert to greyscale
        #     X = transforms.Grayscale()(X)
        # if img_action == 1: # shuffle colour channels
        #     s = [0, 1, 2]
        #     random.shuffle(s)
        #     #X = torch.permute(X, s)
        # if img_action == 2:
        #     X = transforms.ColorJitter()(X)

        # https://pytorch.org/vision/stable/transforms.html
        xforms = [
            RandomColourShuffle(),
            transforms.ColorJitter(),
            transforms.GaussianBlur(9),
            transforms.RandomInvert(1),
            #transforms.RandomPosterize(5, 0.5),
            transforms.RandomSolarize(0.5, 1),
            transforms.RandomAdjustSharpness(0.5, 1),
            transforms.RandomAutocontrast(0.5),
            #transforms.RandomEqualize(0.5)
        ]
        t = transforms.RandomApply(xforms, 0.5)
        X = t(X)

        
        # # convert CSS properties to a set of labels
        # node_cls = 'node' if 'children' in js and len(js['children']) > 0 else 'leaf'
        # display_cls = 'column'
        
        # try:
        #     if js['css']['display'] == 'flex' and js['css']['flex-direction'] == 'row':
        #         display_cls = 'row'
        # except:
        #     pass

        size = [js['bounds']['width'], js['bounds']['height']]
        layout = js['layout']
        first_child_size = [js['first_child_size']['width'], js['first_child_size']['height']]
        first_child_size = [first_child_size[0] / size[0], first_child_size[1] / size[1]] # convert to fraction of parent size


        return X, self.transformer.encode_layout_class(layout), self.transformer.encode_first_child_size(first_child_size) 
        #self.transformer.encode_bounds(parent_wh, bounds_arr), self.transformer.encode_node_class(node_cls), self.transformer.encode_display_class(display_cls) 
