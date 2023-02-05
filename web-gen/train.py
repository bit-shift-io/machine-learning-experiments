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

def main():
    tr = Transformer(image_size=image_size)
    ds = WebsitesDataset('data', transformer=tr)
    train_data, test_data = torch.utils.data.random_split(ds, [int(train_pct * len(ds)), len(ds) - int(train_pct * len(ds))])

    # https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/
    # https://pytorch.org/docs/stable/data.html
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    train_images, train_layout, train_first_child_size = next(iter(train_dataloader))
    subplots = create_subplots(len(train_images))

    torch.backends.cudnn.benchmark = True

    #instantiate CNN model
    model = CNN(image_size=tr.input_size(), out_features=tr.output_size()).to(device)
    print(model)

    def my_loss(output, target):
        loss = torch.mean((output - target)**2)
        return loss

    criterion_first_child_size = torch.nn.MSELoss() #my_loss #
    criterion_layout = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

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

    pbar = tqdm(range(training_epochs))
    for epoch in pbar:
        avg_loss_layout = 0
        avg_loss_first_child_size = 0
        for i, (X, Y_layout, Y_first_child_size) in tqdm(enumerate(train_dataloader), leave=False, total=total_batch):
            optimizer.zero_grad(set_to_none=True) # <= initialization of the gradients
            
            # forward propagation
            # TODO: investigate this: https://pytorch.org/vision/stable/generated/torchvision.ops.generalized_box_iou_loss.html
            # Casts operations to mixed precision
            #with torch.cuda.amp.autocast():
            pred_layout, pred_first_child_size = model(X.to(device, non_blocking=True))

            with torch.no_grad():
                # only show for first batch in the epoch so we don't slow thing too much
                #if i == 0 and (epoch % 20 == 0):
                #    p_size = pred_first_child_size.cpu().detach() #pred_first_child_size[sample_idx].detach().numpy()
                #    show_data_grid(subplots, X.cpu(), Y_first_child_size.cpu(), p_size)
                
                # for rows, we only care about measuring loss on the the x-axis component, so set the y-axis to same as target data
                # for columns we only care about measuing loss on the y-axis component, so set the x-asi the same as the target data
                for i, layout_oh in enumerate(Y_layout):
                    layout = tr.decode_layout_class(layout_oh)
                    if (layout == 'row'):
                        pred_first_child_size[i][1] = Y_first_child_size[i][1]
                    elif (layout == 'column'):
                        pred_first_child_size[i][0] = Y_first_child_size[i][0]

            # https://discuss.pytorch.org/t/is-there-a-way-to-combine-classification-and-regression-in-single-model/165549/2
            # TODO: https://discuss.pytorch.org/t/ignore-loss-on-some-outputs-depending-on-others/170864 
            loss_first_child_size = criterion_first_child_size(pred_first_child_size, Y_first_child_size.to(device, non_blocking=True))
            loss_layout = criterion_layout(pred_layout, Y_layout.to(device, non_blocking=True))

            loss_first_child_size = loss_first_child_size #/ 10.0
            loss_total = loss_first_child_size + loss_layout

            # Backward propagation
            loss_total.backward() # <= compute the gradient of the loss/cost function     
            optimizer.step() # <= Update the gradients
                
            with torch.no_grad():
                avg_loss_first_child_size += loss_first_child_size.data / total_batch
                avg_loss_layout += loss_layout.data / total_batch

        print("\033[F [Epoch: {:>4}], mean loss: layout = {:>.6}, size = {:>.6}\n".format(epoch + io_params['epoch'] + 1, avg_loss_layout.item(), avg_loss_first_child_size.item()))
        #pbar.set_postfix({'epoch': epoch + io_params['epoch'] + 1, 'layout loss': avg_loss_layout.item(), 'size loss': avg_loss_first_child_size.item()})

        if (epoch % 10 == 0):
            #print('Saving model')
            save(model_path, model, optimizer, {
                'epoch': epoch + io_params['epoch'] + 1
            })


    print('Training Finished!')


if __name__ == '__main__':
    main()