import torch
from torch.autograd import Variable

class DNN:
    """Deep Neural Nnetwork"""

    def __init__(self, in_features, out_features, hidden_dim, lr) -> None:
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(in_features, hidden_dim),
                        torch.nn.ELU(),
                        torch.nn.Linear(hidden_dim, hidden_dim * 2),
                        torch.nn.ELU(),
                        torch.nn.Linear(hidden_dim * 2, out_features)
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    def save(self, path, other_params={}):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **other_params
        }, path)

    def load(self, path):
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint
        except:
            return None

    # this performs 1 epoch
    def train(self, batch_x, batch_y, epoch=0):
        """Update the weights of the network given a training sample. """

        y_pred = self.model(torch.Tensor(batch_x))
        loss = self.criterion(y_pred, Variable(torch.Tensor(batch_y)))
        self.optimizer.zero_grad(set_to_none=True)
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