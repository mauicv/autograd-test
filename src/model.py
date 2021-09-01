from functions import Mse


class Model:
    def __init__(self):
        self.layers = []
        self.loss_fn = None

    def add_layer(self, layer):
        if self.layers:
            self.layers[-1].next_layer = layer
        self.layers.append(layer)
        if len(self.layers) > 2:
            self.layers[-1].last_layer = self.layers[-2]

    def add_layers(self, layers):
        for layer in layers:
            self.add_layer(layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, y):
        for layer in self.layers:
            x = layer.forward(x)
        loss = self.loss_fn(x, y)
        dl_dx = self.loss_fn.d(x, y)
        grads = self.backward(dl_dx)
        return loss, grads

    def backward(self, dl_dx):
        grads = []
        for layer in self.layers[::-1]:
            dl_dx = layer.backward(dl_dx)
            grads.append((layer.dl_dw, layer.dl_db))
        return grads

    def compile(self, loss=Mse()):
        self.loss_fn = loss
