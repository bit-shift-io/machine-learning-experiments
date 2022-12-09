import random
import numpy as np
import matplotlib.pyplot as plt
import torch

import torch
from torch.autograd import Variable


# model some 2d equation for testing
class SampleSchedule:
    def num_model_outputs(self):
        return 1

    def reward(self, data):
        return data[0] ** 4 - 20 * data[0] ** 2 +  10 * data[0] + 4 + data[0] * 20

    def random_sample(self):
        return [random.uniform(-5.0, 5.0)]


    def random_samples(self, n_samples):
        """Get N Nrandom samples ready for a training a model"""
        x = []
        y = []
        for _ in range(n_samples):
            x_r = self.random_sample()
            x.append(x_r)

            y_r = self.reward(x_r)
            y.append([y_r])

        return x, y


class DNN:
    """Deep Neural Nnetwork"""

    def __init__(self, out_features, in_features=1, hidden_dim=64, lr=0.05) -> None:
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(in_features, 12),
                        torch.nn.ELU(),
                        torch.nn.Linear(12, 9),
                        torch.nn.ELU(),
                        torch.nn.Linear(9, 6),
                        torch.nn.ELU(),
                        torch.nn.Linear(6, 3),
                        torch.nn.ELU(),
                        torch.nn.Linear(3, out_features)
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    # this performs 1 epoch
    def train(self, batch_x, batch_y, epoch=0):
        """Update the weights of the network given a training sample. """

        y_pred = self.model(torch.Tensor(batch_x))
        loss = self.criterion(y_pred, Variable(torch.Tensor(batch_y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update training loss, accuracy, and the number of samples
        # visited
        #trainLoss = loss.item() * len(batch_y)
        #trainAcc = (y_pred.max(1)[1] == batch_y).sum().item()
        #samples = batch_y.size(0)
        #trainTemplate = "epoch {} train loss: {:.3f} train accuracy: {:.3f}"
        #print(trainTemplate.format(epoch + 1, (trainLoss / samples), (trainAcc / samples)))

    def predict(self, inputs=[0]):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(inputs))

    # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
    def freeze(self):
        """Lock the model so not traning will occur"""
        ct = 0
        for child in self.model.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False




schedule = SampleSchedule()

num_model_outputs = schedule.num_model_outputs()
dnn = DNN(num_model_outputs)





# move this to a training algo class...
n_random_episodes = 100
n_episode = 100
epsilon = 0.3
eps_decay = 0.99
memory = []

n_epochs = 200

x, y = schedule.random_samples(n_random_episodes)
for epoch in range(n_epochs):
    dnn.train(x, y, epoch)

y_pred = dnn.predict(x)


#dnn.freeze()

# modify a new DNN
dnn2 = DNN(1)


model2 = torch.nn.Sequential(
    torch.nn.Linear(1, 1),
    #torch.nn.ELU(), # do i need this? it stops the mean changing, so probbsbly no
)


for i in range(0, 9):
    l = dnn.model[i]

    # Freeze the layer
    for param in l.parameters():
        param.requires_grad = False

    model2.append(l)

dnn2.model = model2 #torch.nn.Sequential(model_layers)
dnn2.optimizer = torch.optim.Adam(dnn2.model.parameters(), 0.05)

#dnn2 = dnn

#y_pred_2 = dnn2.predict(x) # make sure model extends the other model - yes!

# now setup a custom loss function
class CustomLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_true, smooth=1):
        m = torch.mean(y_pred) 
        print(f"mean: {m}") # why is this not changing
        return m

dnn2.criterion = CustomLoss()

# I don't really care about the input and output at this point 
sample_input = np.ones(10000).reshape(10000, 1)
#this doesn't matter at all, we're just going to minimize the function
sample_output = np.array([0] * 10000).reshape(10000, 1)

# train the new DNN
n_epochs = 1000
for epoch in range(n_epochs):
    dnn2.train(sample_input, sample_output, epoch)

y_pred_2 = dnn.predict(x) # make sure the origional model is still unchanged
#y_pred_3 = dnn2.predict(x)

optimum_x = dnn2.model[0].weight.item() # get weight from the first Linear layer
print(optimum_x)
optimum_y = dnn2.predict([optimum_x]).item()


def plot():
    plt.axvline(x=optimum_x)
    plt.axhline(y=optimum_y)

    plt.scatter(x, y, label='Actual', s=2)
    plt.scatter(x, y_pred, color='red', label='Predicted', s=2)
    plt.legend()

plot()

print('Done')