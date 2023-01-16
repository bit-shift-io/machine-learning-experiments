import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import glob
import json
from PIL import Image
import torchvision.transforms.functional as FT
from torchvision import transforms
import torch
from utils import *
from pathlib import Path

# sample: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py
class WebsitesDataset(Dataset):
    def __init__(self, data_dir, image_size):
        search = f'{data_dir}/**/*.json'
        self.samples = glob.glob(search, recursive = True)
        self.image_size = image_size
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with open(sample, 'r') as f:
            js = json.load(f)

        #path = Path(sample)
        #parent_dir = path.parents[1]
        #parent_sample = parent_dir.joinpath('data.json')

        #with open(parent_sample, 'r') as f:
        #    parent_js = json.load(f)


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
        frac_bounds_arr = to_fractional_scale(bounds_arr, parent_wh)
        centre_bounds_arr = xy_to_cxcy(frac_bounds_arr)
        bounds = torch.FloatTensor(centre_bounds_arr)

        # the sample code above applies random variation and flips etc...
        # do we need to do something similar to help AI in fuzzy situations?
        image = Image.open(js['img_path']).convert("RGB")

        #pil_to_tensor = transforms.ToTensor()(image).unsqueeze_(0)
        #print(pil_to_tensor.shape) 

        image = FT.pil_to_tensor(image)
        #im = transforms.ToPILImage()(image).convert("RGB")

        #image = FT.pil_to_tensor(image)
        image = FT.resize(image, self.image_size)

        # TODO: massage js into a label

        #if self.transform:
        #    image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        label = 'test'
        return image, bounds