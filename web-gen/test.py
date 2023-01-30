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
from config import *

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

layout_acc_arr = []
size_acc_arr = []

with torch.no_grad():
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    for i, (X, Y_layout, Y_first_child_size) in tqdm(enumerate(test_dataloader), leave=False):
        pred_layout, pred_first_child_size = model(X)

        # layout accuracy
        for i, (pred, actual) in enumerate(zip(pred_layout, Y_layout)):            
            dec_pred_layout = tr.decode_layout_class(pred)
            dec_actual_layout = tr.decode_layout_class(actual)

            layout_acc = (dec_pred_layout == dec_actual_layout)
            layout_acc_arr.append(layout_acc)

        # accuracy for size
        for i, (pred, actual) in enumerate(zip(pred_first_child_size, Y_first_child_size)):
            layout_acc = (dec_pred_layout == dec_actual_layout)
            layout_acc_arr.append(layout_acc)

            layout = tr.decode_layout_class(Y_layout[i])
            if layout == 'row':
                size_acc = 1.0 - abs(pred[0] - actual[0])
            elif layout == 'column':
                size_acc = 1.0 - abs(pred[1] - actual[1])
            
            #box1 = torch.tensor([cxcy_to_box(cxcy1)], dtype=torch.float)
            # iou = bops.box_iou(box1, box2).item()
            size_acc_arr.append(size_acc)



avg_layout_acc = np.array(layout_acc_arr).mean()
avg_size_acc = np.array(size_acc_arr).mean()

print('\nAccuracy for Layout: {:2.2f} %'.format(avg_layout_acc*100))
print('Accuracy for Size: {:2.2f} %'.format(avg_size_acc*100))
print('\n')
