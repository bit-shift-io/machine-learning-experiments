import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import glob
import json
from PIL import Image
import torchvision.transforms.functional as FT
from torchvision import transforms

# sample: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py
class WebsitesDataset(Dataset):
    def __init__(self, data_dir):
        search = f'{data_dir}/**/*.json'
        self.samples = glob.glob(search, recursive = True)
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with open(sample, 'r') as f:
            js = json.load(f)


        # the sample code above applies random variation and flips etc...
        # do we need to do something similar to help AI in fuzzy situations?
        image = Image.open(js['img_path']).convert("RGB")

        #pil_to_tensor = transforms.ToTensor()(image).unsqueeze_(0)
        #print(pil_to_tensor.shape) 

        image = FT.pil_to_tensor(image)
        #im = transforms.ToPILImage()(image).convert("RGB")

        #image = FT.pil_to_tensor(image)
        image = FT.resize(image, [128, 128])

        # TODO: massage js into a label

        #if self.transform:
        #    image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        label = 'test'
        return image, label