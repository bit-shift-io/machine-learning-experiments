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


# sample: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py
class WebsitesDataset(Dataset):
    def __init__(self, data_dir, transformer):
        search = f'{data_dir}/**/*.json'
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

        parent_wh = [js['parent_size']['width'], js['parent_size']['height']] #[parent_js['bounds']['width'], parent_js['bounds']['height']]
        bounds_arr = [js['bounds']['x'], js['bounds']['y'], js['bounds']['width'], js['bounds']['height']]
        

        # the sample code above applies random variation and flips etc...
        # do we need to do something similar to help AI in fuzzy situations?
        image = Image.open(js['img_path'])
        
        # convert CSS properties to a set of labels
        node_cls = 'node' if 'children' in js and len(js['children']) > 0 else 'leaf'
        display_cls = 'column'

        try:
            if js['css']['display'] == 'flex' and js['css']['flex-direction'] == 'row':
                display_cls = 'row'
        except:
            pass

        return self.transformer.encode_inputs(image), self.transformer.encode_outputs(parent_wh, bounds_arr, node_cls, display_cls)
