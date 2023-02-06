# https://machinelearningknowledge.ai/pytorch-conv2d-explained-with-examples/

import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn.init
import matplotlib.pyplot as plt
from datasets import WebsitesDataset
from transformer import Transformer
from tqdm import tqdm
from model_io import save, load
from model import CNN
from config import *
from utils import *
import numpy as np
import random
from debug import *

batch_size = 16

tr = Transformer(image_size=image_size)
ds = WebsitesDataset('data', transformer=tr, debug=True)
train_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

subplots = create_subplots(batch_size)

total_batch = len(ds) / batch_size

print(f'\nBatch 0')
for i, (X, Y_layout, Y_first_child_size) in enumerate(train_dataloader):
    show_data_grid(subplots, X, Y_first_child_size)
    plt.pause(10)
    print(f'\nBatch {i+1}')

print('Finished!')
