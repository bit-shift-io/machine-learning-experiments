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
from model import CNN2
from config import *
from utils import *
import numpy as np
import random
from debug import *

tr = Transformer(image_size=image_size)
ds = WebsitesDataset('data', transformer=tr)
train_data, test_data = torch.utils.data.random_split(ds, [int(train_pct * len(ds)), len(ds) - int(train_pct * len(ds))])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

train_images, train_layout, train_first_child_size = next(iter(train_dataloader))
subplots = create_subplots(train_images)


#instantiate CNN model
model = CNN2(image_size=tr.input_size(), out_features=tr.output_size())
print(model)

criterion_first_child_size = torch.nn.MSELoss()
criterion_layout = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# load existing model
io_params = load(model_path, model, optimizer, {
    'epoch': 0
})

#print(model.bounds_out[-4].weight)


model.train()

print('Training the Deep Learning network ...')
train_loss = []
train_accu = []

total_batch = len(train_data) / batch_size

print('Size of the training dataset is {}'.format(len(train_data)))
print('Size of the testing dataset is {}'.format(len(test_data)))
print('Batch size is : {}'.format(batch_size))
print('Total number of batches is : {0:2.0f}'.format(total_batch))
print('\nTotal number of epochs is : {0:2.0f}'.format(training_epochs))

for epoch in tqdm(range(training_epochs)):
    avg_loss_layout = 0
    avg_loss_first_child_size = 0
    for i, (X, Y_layout, Y_first_child_size) in tqdm(enumerate(train_dataloader), leave=False, total=total_batch):
        optimizer.zero_grad() # <= initialization of the gradients
        
        # forward propagation
        # TODO: investigate this: https://pytorch.org/vision/stable/generated/torchvision.ops.generalized_box_iou_loss.html
        pred_layout, pred_first_child_size = model(X)

        # help us debug the data
        #sample_idx = random.randint(0, Y_layout.shape[0] - 1)

        # only show for first batch in the epoch so we don't slow thing too much
        if i == 0:
            p_size = pred_first_child_size.detach() #pred_first_child_size[sample_idx].detach().numpy()
            show_data_grid(subplots, X, Y_first_child_size, p_size)

        # testing - just to help test the decoder outputs code
        #decoded_pred = tr.decode_output(hypothesis)

        # TODO: do I need this bit below? as we now return a full image bounds for leaf nodes, which might even help the NN learn bounds 
        #for i, (pred_n, y_n) in enumerate(zip(pred_node, Y_node_class)):
        #    y_node_class = tr.decode_node_class(y_n)
        #    if y_node_class == 'leaf':
        #        pred_bounds[i] = Y_bounds[i] # set loss to zero for this case, as we don't care about bounding boxes for leaf nodes


        # https://discuss.pytorch.org/t/is-there-a-way-to-combine-classification-and-regression-in-single-model/165549/2
        # TODO: https://discuss.pytorch.org/t/ignore-loss-on-some-outputs-depending-on-others/170864 
        loss_first_child_size = criterion_first_child_size(pred_first_child_size, Y_first_child_size)
        loss_layout = criterion_layout(pred_layout, Y_layout)

        loss_first_child_size = loss_first_child_size / 10.0
        loss_total = loss_first_child_size + loss_layout

        # Backward propagation
        loss_total.backward() # <= compute the gradient of the loss/cost function     
        optimizer.step() # <= Update the gradients
             

            
        # Print some performance to monitor the training
        #prediction = hypothesis.data.max(dim=1)[1]
        #train_accu.append(((prediction.data == Y.data).float().mean()).item())

        # just compute accuracy for the
        #pred_0 = X[0]
        #y_0 = Y_bounds[0]
        #bounds_acc = cxcy_to_iou(pred_bounds, actual_bounds)
        #node_cls_acc = (pred[1] == actual[1])
        #display_cls_acc = (pred[2] == actual[2])

        #train_loss.append(loss_total.item())   
        #if i % 200 == 0:
        #    print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch+1, i, train_loss[-1], train_accu[-1]))
       
        avg_loss_first_child_size += loss_first_child_size.data / total_batch
        avg_loss_layout += loss_layout.data / total_batch

    print("[Epoch: {:>4}], mean loss: layout = {:>.9}, first child sz = {:>.9}".format(epoch + io_params['epoch'] + 1, avg_loss_layout.item(), avg_loss_first_child_size.item()))
    
    #print(model.bounds_out[-4].weight)

    save(model_path, model, optimizer, {
        'epoch': epoch + io_params['epoch'] + 1
    })

print('Training Finished!')

#from matplotlib import pylab as plt
#import numpy as np
#plt.figure(figsize=(20,10))
#plt.subplot(121), plt.plot(np.arange(len(train_loss)), train_loss), plt.ylim([0,10])
#plt.subplot(122), plt.plot(np.arange(len(train_accu)), 100 * torch.as_tensor(train_accu).numpy()), plt.ylim([0,100])

