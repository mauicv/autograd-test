import unittest
import numpy as np
import torch
from torch import nn
from torch_net import NeuralNetwork
from src.model import Model
from src.layer import Layer
from src.functions import Linear
# from util import print_layer_weights


class ReverseModeTests(unittest.TestCase):
    def test_correct_derivative_simple(self):
        input = [1.0, 1.0, 1.0]
        output = [1.0, 1.0, 1.0]
        layer_dims = [3, 3, 3, 3, 3]

        X = np.array(input)
        Y = np.array(output)
        X_t = torch.tensor(X)
        Y_t = torch.tensor(Y)

        # Model:
        model = Model()
        layers = [Layer(d1, d2) for d1, d2 in zip(layer_dims, layer_dims[2:])]
        last_layer = Layer(layer_dims[-2], layer_dims[-1], activation=Linear())
        layers.append(last_layer)

        torch_model = NeuralNetwork()
        torch_model.add_layers([(l.W, l.b) for l in layers])

        model.add_layers(layers)
        model.compile()

        torch_out = torch_model(X_t)
        loss = nn.MSELoss()
        torch_loss = loss(torch_out, Y_t)
        torch_loss.backward()

        torch_numpy_out = torch_out.detach().numpy()
        model_out = model(X)
        self.assertEqual(torch_numpy_out.shape, model_out.shape)
        np.testing.assert_almost_equal(torch_numpy_out, model_out)
        loss, grads = model.forward(X, Y)
        self.assertEqual(round(loss, 5), round(torch_loss.tolist(), 5))

        # print_layer_weights(torch_model.layers, layers)

        for i, ((W, b), layer) in \
                enumerate(zip(torch_model.layers, layers)):
            torch_dw = W.grad.detach().numpy()
            torch_db = b.grad.detach().numpy()
            np.testing.assert_almost_equal(layer.dl_dw, torch_dw)
            np.testing.assert_almost_equal(layer.dl_db, torch_db)
            self.assertEqual(layer.dl_dw.shape, torch_dw.shape)
            self.assertEqual(layer.dl_db.shape, torch_db.shape)


if __name__ == '__main__':
    unittest.main()
