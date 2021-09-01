import unittest
import numpy as np
import torch
from torch import nn
from torch_net import NeuralNetwork
from src.model import Model
from src.layer import Layer
from src.functions import Linear


class ReverseModeTests(unittest.TestCase):
    def test_correct_derivative(self):
        torch_model = NeuralNetwork()
        X = torch.tensor([[1., 1.]])
        Y = torch.tensor([[0., 0.]])
        torch_out = torch_model(X)
        loss = nn.MSELoss()
        torch_loss = loss(torch_out, Y)
        torch_loss.backward()

        # for w, b in [torch_model.l1.weight.grad,
        #              torch_model.l1.bias.grad,
        #              torch_model.l2.weight.grad,
        #              torch_model.l2.bias.grad,
        #              torch_model.o.weight.grad,
        #              torch_model.o.bias.grad]:
        #     print(w.tolist(), b.tolist())

        # Model:
        model = Model()
        layers = [Layer(2, 2) for _ in range(2)]
        output_layer = [Layer(2, 2, activation=Linear())]
        layers = layers + output_layer
        layers[0].W = torch_model.l1.weight.detach().numpy()
        layers[0].b = torch_model.l1.bias.detach().numpy()
        layers[1].W = torch_model.l2.weight.detach().numpy()
        layers[1].b = torch_model.l2.bias.detach().numpy()
        layers[2].W = torch_model.o.weight.detach().numpy()
        layers[2].b = torch_model.o.bias.detach().numpy()
        model.add_layers(layers)
        model.compile()

        x = np.ones(2)
        y = np.zeros(2)

        for a, b in zip(torch_out.tolist()[0], model(x)):
            self.assertEqual(round(a, 5), round(b, 5))

        loss, grads = model.forward(x, y)

        self.assertEqual(round(loss, 5), round(torch_loss.tolist(), 5))

        # for w, b in grads:
        #     print(w.tolist(), b.tolist())


if __name__ == '__main__':
    unittest.main()
