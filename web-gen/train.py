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

tr = Transformer(image_size=image_size)
ds = WebsitesDataset('data', transformer=tr)
train_data, test_data = torch.utils.data.random_split(ds, [int(train_pct * len(ds)), len(ds) - int(train_pct * len(ds))])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# https://jovian.ml/aakanksha-ns/road-signs-bounding-box-prediction
def create_corner_rect(cxcy, color='red'):
    xy = cxcy_to_xy(cxcy)
    xy = from_fractional_scale(xy, image_size)
    rect = plt.Rectangle((xy[0], xy[1]), xy[2], xy[3], edgecolor=color, facecolor='none', lw=3)
    return rect

# Display image and label.
def showimg():
    train_images, train_bounds, train_node_class, train_display_class = next(iter(train_dataloader))
    img = train_images[0]#.squeeze()
    plt.imshow(img.permute(1, 2, 0))
    plt.gca().add_patch(create_corner_rect(train_bounds[0]))
    plt.show()

showimg()


#instantiate CNN model
model = CNN2(image_size=tr.input_size(), out_features=tr.output_size())
print(model)

criterion_1 = torch.nn.MSELoss()
criterion_2 = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
criterion_3 = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# load existing model
io_params = load(model_path, model, optimizer, {
    'epoch': 0
})

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
    avg_loss = 0
    for i, (X, Y_bounds, Y_node_class, Y_display_class) in tqdm(enumerate(train_dataloader), leave=False, total=total_batch):
        optimizer.zero_grad() # <= initialization of the gradients
        
        # forward propagation
        # TODO: investigate this: https://pytorch.org/vision/stable/generated/torchvision.ops.generalized_box_iou_loss.html
        pred_bounds, pred_node, pred_display = model(X)

        # testing - just to help test the decoder outputs code
        #decoded_pred = tr.decode_output(hypothesis)

        # https://discuss.pytorch.org/t/is-there-a-way-to-combine-classification-and-regression-in-single-model/165549/2
        loss_1 = criterion_1(pred_bounds, Y_bounds)
        loss_2 = criterion_2(pred_node, Y_node_class)
        loss_3 = criterion_3(pred_display, Y_display_class)
        #loss_total = reg_weight * loss_regression + class_1_weight * loss_classification_1 + class_2_weight * loss_classification_2
        loss_total = loss_1/1000.0 + loss_2 + loss_3

        #cost = criterion(hypothesis, Y) # <= compute the loss function
        
        # Backward propagation
        loss_total.backward() #cost.backward() # <= compute the gradient of the loss/cost function     
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

        train_loss.append(loss_total.item())   
        #if i % 200 == 0:
        #    print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch+1, i, train_loss[-1], train_accu[-1]))
       
        avg_loss += loss_total.data / total_batch

    print("[Epoch: {:>4}], averaged loss = {:>.9}".format(epoch + io_params['epoch'] + 1, avg_loss.item()))
    save(model_path, model, optimizer, {
        'epoch': epoch + io_params['epoch'] + 1
    })


print('Learning Finished!')

from matplotlib import pylab as plt
import numpy as np
plt.figure(figsize=(20,10))
plt.subplot(121), plt.plot(np.arange(len(train_loss)), train_loss), plt.ylim([0,10])
plt.subplot(122), plt.plot(np.arange(len(train_accu)), 100 * torch.as_tensor(train_accu).numpy()), plt.ylim([0,100])

