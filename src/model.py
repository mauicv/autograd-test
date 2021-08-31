import numpy as np
from layer import Layer


class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        if self.layers:
            self.layers[-1].next_layer = layer
        self.layers.append(layer)
        if len(self.layers) > 2:
            self.layers[-1].last_layer = self.layers[-2]

    def add_layers(self, layers):
        for layer in layers:
            self.add_layer(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self):
        # fails becuase last row hasn't computed dl_dx yet
        for layer in self.layers[::-1]:
            layer.backward()


if __name__ == "__main__":
    model = Model()
    layer1 = Layer(10, 10)
    layer2 = Layer(10, 10)
    layer3 = Layer(10, 10)
    layer4 = Layer(10, 10)

    model.add_layers([layer1, layer2, layer3, layer4])
    layer = layer1
    print(model.forward(np.ones(10)))
    model.backward()
