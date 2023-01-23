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
from utils import cxcy_to_iou
import numpy as np
from model import CNN2

model_path = 'model.pt'

# hyperparameters
batch_size = 32

keep_prob = 0.8
hidden_sz = 1024 #out_features * out_features # TODO: make this a hyper param?

image_size=[100, 100]
reg_weight = 0.3333
class_1_weight = 0.3333
class_2_weight = 0.3333

learning_rate = 0.001

train_pct = 0.8 #0.001 # should e about 0.8, reduce to lower to speed up training for testing only
training_epochs = 100 # should be abbout 100, reduce to speed up testing

tr = Transformer(image_size=image_size)
ds = WebsitesDataset('data', transformer=tr)
train_data, test_data = torch.utils.data.random_split(ds, [int(train_pct * len(ds)), len(ds) - int(train_pct * len(ds))])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


#instantiate CNN model
model = CNN2(image_size=tr.input_size(), out_features=tr.output_size())
print(model)

# load existing model
io_params = load(model_path, model, None, {
    'epoch': 0
})

# Test model and check accuracy
model.eval()    # set the model to evaluation mode (dropout=False)

bounds_acc_arr = []
node_cls_acc_arr = []
display_cls_acc_arr = []

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
for i, (batch_X, batch_Y) in tqdm(enumerate(train_dataloader)):
    X = batch_X    # images
    Y = batch_Y    # labels are not one-hot encoded

    prediction = model(X)

    decoded_pred = tr.decode_output(prediction)
    decoded_actual = tr.decode_output(Y)

    for i, (pred, actual) in enumerate(zip(decoded_pred, decoded_actual)):
        pred_bounds = pred[0] # cxcy
        actual_bounds = actual[0] # cxcy

        bounds_acc = cxcy_to_iou(pred_bounds, actual_bounds)
        node_cls_acc = (pred[1] == actual[1])
        display_cls_acc = (pred[2] == actual[2])

        bounds_acc_arr.append(bounds_acc)
        node_cls_acc_arr.append(float(node_cls_acc))
        display_cls_acc_arr.append(float(display_cls_acc))
        pass



avg_bounds_acc = np.array(bounds_acc_arr).mean()
avg_node_cls_acc = np.array(node_cls_acc_arr).mean()
avg_display_cls_acc = np.array(display_cls_acc_arr).mean()

print('\nAccuracy for Bounds: {:2.2f} %'.format(avg_bounds_acc*100))
print('\nAccuracy for Node Class: {:2.2f} %'.format(avg_node_cls_acc*100))
print('\nAccuracy for Display Class: {:2.2f} %'.format(avg_display_cls_acc*100))
