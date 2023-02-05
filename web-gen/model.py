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
from torchshape import tensorshape

class CNN(torch.nn.Module):
    def __init__(self, image_size, out_features):
        super(CNN, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=conv_sz, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #torch.nn.Dropout(p=1-keep_prob),

            torch.nn.Conv2d(in_channels=conv_sz, out_channels=conv_sz, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob),

            #torch.nn.Conv2d(in_channels=conv_sz, out_channels=conv_sz, kernel_size=(3, 3), padding=1),
            #torch.nn.ReLU(),
            #torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #torch.nn.Dropout(p=1-keep_prob)
        )

        # compute the output size for the above Sequential 
        outshape = [1, 1] + image_size #(batch_size, in_channels, image_height, image_width)
        for module in self.conv.children():
            outshape = tensorshape(module, outshape)
        conv_output_size = outshape[1] * outshape[2] * outshape[3]

        self.size_out = torch.nn.Sequential(
            # todo, convert image to grayscale, conv2d -> single colour channel. Colour shouldnt play a factor in the bounding box
            torch.nn.Flatten(),
            torch.nn.Dropout(p=1-keep_prob),
            torch.nn.Linear(conv_output_size, hidden_sz),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-keep_prob),

            torch.nn.Linear(hidden_sz, size_len)
        )

        self.classifier_features = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(p=1-keep_prob),
            torch.nn.Linear(conv_output_size, hidden_sz),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-keep_prob),
        )

        # self.node_class_out = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_sz, node_classes_len)
        # )

        self.layout_class_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_sz, display_classes_len)
        )

    def forward(self, x):
        # conv -> bounds_out
        # conv -> classifier_feautures -> node_classes_out
        # conv -> classifier_feautures -> display_classes_out
        t = self.conv(x)
        b = self.size_out(t)

        u = self.classifier_features(t)
        l = self.layout_class_out(u)
        
        return l, b