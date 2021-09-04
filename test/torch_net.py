# import os
import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu


class NeuralNetwork(nn.Module):
    def __init__(self, layer_dims=None):
        super(NeuralNetwork, self).__init__()
        self.layers = []

    def add_layers(self, layers):
        for W, b in layers:
            self.add_layer(W, b)

    def add_layer(self, weights, biases):
        W = torch.tensor(weights, requires_grad=True)
        b = torch.tensor(biases, requires_grad=True)
        self.layers.append((W, b))

    def forward(self, x):
        x = x.double()
        for W, b in self.layers[:-1]:
            x = relu(x @ W.T + b)
        W, b = self.layers[-1]
        return x @ W.T + b


if __name__ == "__main__":
    model = NeuralNetwork()
    model.add_layer(
        weights=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        biases=np.array([1.0, 1.0, 1.0])
    )
    o = model(torch.tensor([1.0, 1.0]))
    print(o)
