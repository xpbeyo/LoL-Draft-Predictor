import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch

import matplotlib.pyplot as plt
#import panda as pd


def reward(state, a):
    return


def new_Q(state, a, lr, d):
    #New Q(s, a) = Q(s, a) + lr [R(s,a) + d*maxQ'(s', a') - Q(s, a)]
    return


def sigmoid(x):
    """Uses sigmoid function on <x>"""
    return np.exp(x) / (1 + np.exp(x))

class Winrate_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size=5):
        """ Initialize a class Winrate_Predictor.
        :param input_size: int
        :param hidden_size: int
        """
        super(Winrate_Predictor, self).__init__()

        # The number of hidden units for each layer
        # = 150
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define linear functions.
        self.g = nn.Linear(self.input_size, self.hidden_size)
        self.h = nn.Linear(self.hidden_size, 2)
        self.s = nn.Softmax()
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
        out = self.r(out)
        out = self.h(out)
        out = self.s(out)

        return out


def train(model, num_epoch, data, lr, batch_size, discount):
    model.train()

    inputs = data[:, 1:].float()
    targets = data[:, 0].long()
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size)

    # Define optimizers and loss function.
    optimizer = optim.ASGD(model.parameters(), lr=lr)

    CELoss = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(0, num_epoch):
        loss_epoch = 0.
        for sample in loader:
            inputs_mini = sample[0]
            targets_mini = sample[1]

            prediction = model(inputs_mini)

            loss = CELoss(prediction, targets_mini)
            optimizer.zero_grad()
            loss.backward()
            # print(model.g.weight.grad)
            optimizer.step()

            loss_epoch += loss.item()

        normalized_loss = loss_epoch / inputs.size(0)
        losses.append(normalized_loss)
        print("Epoch: {} \tTraining Cost: {:.6f}\t".format(epoch, normalized_loss))

        acc = torch.sum(torch.argmax(model(inputs), dim=1).squeeze() == targets) / inputs.size(0)
        print("Epoch: {} \tTraining Accuracy: {:.6f}\t".format(epoch, acc))

    plt.plot(range(1, num_epoch + 1), losses, label="Training Loss")
    plt.title(f"Training Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

def load_data(path):
    return torch.from_numpy(np.loadtxt(path, delimiter=",", skiprows=1))

def main():
    data = load_data("./team_comp.csv")

    torch.manual_seed(11)
    model = Winrate_Predictor(input_size=data.size(1) - 1, hidden_size=100)

    # Optimization hyperparameters.
    lr = 0.005
    num_epoch = 1000
    bs = 16

    #Do we want a lambda for loss?
    discount = None

    train(model, num_epoch, data, lr, bs, discount)
    torch.save(model, "winrate_predictor.pt")

if __name__ == "__main__":
    main()
