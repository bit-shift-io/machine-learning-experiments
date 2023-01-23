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
from config import *

# Implementation of CNN/ConvNet Model
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        # Conv -> (?, 28, 28, 32)
        # Pool -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 14, 14, 32)
        # Conv      ->(?, 14, 14, 64)
        # Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        # Conv ->(?, 7, 7, 128)
        # Pool ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class CNN2(torch.nn.Module):
    def __init__(self, image_size, out_features):
        super(CNN2, self).__init__()
        
        #total_sz = image_size[0] * image_size[1] * 3 #image_size[3]
        #max_pool_sz_1 = int(total_sz / 4)
        #max_pool_sz_2 = int(max_pool_sz_1 / 4)

        conv_output_size = 7500 # can we compute this from image_size?

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob),

            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob)
        )

        self.bounds_out = torch.nn.Sequential(
            # todo, convert image to grayscale, conv2d -> single colour channel. Colour shouldnt play a factor in the bounding box
            torch.nn.Flatten(),
            torch.nn.Linear(conv_output_size, hidden_sz),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-keep_prob),

            torch.nn.Linear(hidden_sz, 4)
        )

        self.classifier_features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(conv_output_size, hidden_sz),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-keep_prob),
        )

        self.node_class_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_sz, node_classes_len)
        )

        self.display_class_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_sz, display_classes_len)
        )

    def forward(self, x):
        # conv -> bounds_out
        # conv -> classifier_feautures -> node_classes_out
        # conv -> classifier_feautures -> display_classes_out
        t = self.conv(x)
        b = self.bounds_out(t)

        u = self.classifier_features(t)
        n = self.node_class_out(u)
        d = self.display_class_out(u)
        

        return b, n, d