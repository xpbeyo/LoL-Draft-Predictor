from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self):
        """ Initialize a class AutoEncoder.
        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # The number of hidden units for each layer
        layer1 = 5
        # = 150

        # Define linear functions.
        self.g = nn.Linear(10, layer1)
        self.h = nn.Linear(layer1, 1)
        self.s = nn.Sigmoid()
        self.r = nn.ReLU()

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.
        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.
        :param inputs: user vector.
        :return: user vector.
        """
        out = self.g(inputs)
        out = self.s(out)
        out = self.r(out)
        out = self.s(out)
        out = self.h(out)

        return out


def train(model, num_epoch, lr, lamb, data):
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []

    for epoch in range(0, num_epoch):
        loss = 0.

        for stage in range(0, 10):
            inputs = Variable(data[stage]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            loss = torch.sum((output - target) ** 2.) + lamb / 2 * (
                        model.get_weight_norm() ** 2)
            loss.backward()

            loss += loss.item()
            optimizer.step()

        losses.append(loss)
        print("Epoch: {} \tTraining Cost: {:.6f}\t".format(epoch, loss))

    plt.plot(range(1, num_epoch + 1), losses, label="Training Loss")
    plt.title(f"Training Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

def load_data(path=""):
    return

def main():
    data = load_data()

    model = AutoEncoder()
    # Optimization hyperparameters.
    lr = 0.005
    num_epoch = 50

    #Do we want a lambda for loss?
    lamb = None

    train(model, num_epoch, lr, lamb, data)


if __name__ == "__main__":
    main()
