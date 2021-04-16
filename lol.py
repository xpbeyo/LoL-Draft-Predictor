from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

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


def train(model, num_epoch, lr, lamb):
    model.train()

    for epoch in range(0, num_epoch):
    return

def load_data(path=""):
    return

def main():
    data = load_data()

    model = AutoEncoder()
    # Optimization hyperparameters.
    lr = None
    num_epoch = None
    lamb = None

    train(model, num_epoch, lr, lamb)


if __name__ == "__main__":
    main()
