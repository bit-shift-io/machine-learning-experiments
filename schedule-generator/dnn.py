import torch
from torch.autograd import Variable

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