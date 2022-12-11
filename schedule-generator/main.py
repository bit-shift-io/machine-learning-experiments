import rules
from schedule import Schedule, SampleSchedule
from dnn import DNN
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

#schedule = Schedule()
#schedule.load('schedule.json')

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


print('qwe')

# this loss function is just the value of the function we're trying to minimize.
# all of the inputs are the same and so we don't really care about y_true
#def my_loss_fn(y_true, y_pred):
#    return tf.reduce_mean(y_pred, axis=-1) 

# check we have learned
#pred_y = dnn.predict([0])
#actual_y = schedule.reward([0])
#accuracy = actual_y - pred_y[0]
#print("Accuracy of model:", accuracy)

# now try to learn these samples
#for episode in range(n_episode):
#    if random.random() < epsilon:
#        output = schedule.random_sample()
#    else:
#        output = dnn.predict()
#
#    reward = schedule.reward(output)
#
#    # How do we maximize reward if we dont know state -> next_state as required by Q-learnning?

#    memory.append((output, reward))

    # Update epsilon
#    epsilon = max(epsilon * eps_decay, 0.01)

def plot():
    plt.axvline(x=optimum_x)
    plt.axhline(y=optimum_y)

    plt.scatter(x, y, label='Actual', s=2)
    plt.scatter(x, y_pred, color='red', label='Predicted', s=2)
    plt.legend()

plot()

#print(schedule)
print('Done')