# import os
# import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(2, 2)
        self.a2 = nn.ReLU()
        self.o = nn.Linear(2, 2)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.a1(x1)
        x3 = self.l2(x2)
        x4 = self.a2(x3)
        x5 = self.o(x4)
        return x5

    def get_weights(self):
        return [(self.l1.weight.tolist(), self.l1.bias.tolist()),
                (self.l2.weight.tolist(), self.l2.bias.tolist()),
                (self.o.weight.tolist(), self.o.bias.tolist())]
