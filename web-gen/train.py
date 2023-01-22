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


# Display image and label.
def showimg():
    train_features, train_labels = next(iter(train_dataloader))
    img = train_features[0].squeeze()
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

#showimg()

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
        super().__init__()
        
        #total_sz = image_size[0] * image_size[1] * 3 #image_size[3]
        #max_pool_sz_1 = int(total_sz / 4)
        #max_pool_sz_2 = int(max_pool_sz_1 / 4)

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob),

            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1-keep_prob),

            torch.nn.Flatten(),
            torch.nn.Linear(1875, hidden_sz), # max_pool_sz_1
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-keep_prob),

            torch.nn.Linear(hidden_sz, out_features)
        )

    def forward(self, x):
        out = self.main(x)
        return out

#instantiate CNN model
model = CNN2(image_size=tr.input_size(), out_features=tr.output_size())
print(model)


criterion_1 = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
criterion_2 = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# load existing model
io_params = load(model_path, model, optimizer, {
    'epoch': 0
})

print('Training the Deep Learning network ...')
train_cost = []
train_accu = []


total_batch = len(train_data) / batch_size

print('Size of the training dataset is {}'.format(len(train_data)))
print('Size of the testing dataset is {}'.format(len(test_data)))
print('Batch size is : {}'.format(batch_size))
print('Total number of batches is : {0:2.0f}'.format(total_batch))
print('\nTotal number of epochs is : {0:2.0f}'.format(training_epochs))

for epoch in tqdm(range(training_epochs)):
    avg_cost = 0
    for i, (batch_X, batch_Y) in tqdm(enumerate(train_dataloader), leave=False):
        X = batch_X    # images
        Y = batch_Y    # labels are not one-hot encoded

        optimizer.zero_grad() # <= initialization of the gradients
        
        # forward propagation
        hypothesis = model(X)

        # testing - just to help test the decoder outputs code
        #decoded_pred = tr.decode_output(hypothesis)

        # https://discuss.pytorch.org/t/is-there-a-way-to-combine-classification-and-regression-in-single-model/165549/2
        # first 4 params are a regression problem - fit the bounding box
        # the next 2 are for 2 classes
        # the next 2 are another 2 class
        reg_target = Y[:, 0:4]
        class_target_1 = Y[:, 4:6]
        class_target_2 = Y[:, 6:8]
        loss_regression = torch.nn.MSELoss() (hypothesis[:, 0:4], reg_target)
        loss_classification_1 = torch.nn.CrossEntropyLoss() (hypothesis[:, 4:6], class_target_1)
        loss_classification_2 = torch.nn.CrossEntropyLoss() (hypothesis[:, 6:8], class_target_2)
        #loss_total = reg_weight * loss_regression + class_1_weight * loss_classification_1 + class_2_weight * loss_classification_2
        loss_total =  loss_regression +  loss_classification_1 + loss_classification_2

        #cost = criterion(hypothesis, Y) # <= compute the loss function
        
        # Backward propagation
        loss_total.backward() #cost.backward() # <= compute the gradient of the loss/cost function     
        optimizer.step() # <= Update the gradients
             
        # Print some performance to monitor the training
        #prediction = hypothesis.data.max(dim=1)[1]
        #train_accu.append(((prediction.data == Y.data).float().mean()).item())
        train_cost.append(loss_total.item())   
        #if i % 200 == 0:
        #    print("Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(epoch+1, i, train_cost[-1], train_accu[-1]))
       
        avg_cost += loss_total.data / total_batch

    print("[Epoch: {:>4}], averaged loss = {:>.9}".format(epoch + io_params['epoch'] + 1, avg_cost.item()))
    save(model_path, model, optimizer, {
        'epoch': epoch + io_params['epoch']
    })


print('Learning Finished!')

from matplotlib import pylab as plt
import numpy as np
plt.figure(figsize=(20,10))
plt.subplot(121), plt.plot(np.arange(len(train_cost)), train_cost), plt.ylim([0,10])
plt.subplot(122), plt.plot(np.arange(len(train_accu)), 100 * torch.as_tensor(train_accu).numpy()), plt.ylim([0,100])


# Test model and check accuracy
model.eval()    # set the model to evaluation mode (dropout=False)

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
for i, (batch_X, batch_Y) in enumerate(train_dataloader):
    X = batch_X    # images
    Y = batch_Y    # labels are not one-hot encoded

    prediction = model(X)

    decoded_pred = tr.decode_output(prediction)
    decoded_actual = tr.decode_output(Y)
    print(decoded_pred)
    exit()



exit()
X_test = test_data.samples.view(len(test_data), 1, 28, 28).float()
Y_test = test_data.targets

prediction = model(X_test)

# Compute accuracy
correct_prediction = (torch.max(prediction.data, dim=1)[1] == Y_test.data)
accuracy = correct_prediction.float().mean().item()
print('\nAccuracy: {:2.2f} %'.format(accuracy*100))

from matplotlib import pylab as plt

plt.figure(figsize=(15,15), facecolor='white')
for i in torch.arange(0,12):
  val, idx = torch.max(prediction, dim=1)
  plt.subplot(4,4,i+1)  
  plt.imshow(X_test[i][0])
  plt.title('This image contains: {0:>2} '.format(idx[i].item()))
  plt.xticks([]), plt.yticks([])
  plt.plt.subplots_adjust()