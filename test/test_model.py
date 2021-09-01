from src.model import Model
from src.layer import Layer
from src.functions import Linear
import unittest
import numpy as np


class ModelTests(unittest.TestCase):
    def test_create_model(self):
        model = Model()
        layers = [Layer(2, 2) for _ in range(2)]
        output_layer = [Layer(2, 2, activation=Linear())]
        model.add_layers(layers+output_layer)
        model.compile()
        x = np.ones(2)
        y = np.zeros(2)
        loss, grads = model.forward(x, y)
