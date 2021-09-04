import numpy as np
from src.functions import Relu


class Layer:
    def __init__(self,
                 input_shape,
                 output_shape,
                 activation=Relu()):

        self.a = activation
        self.W = np.random.normal(0, 0.2, (output_shape, input_shape))
        self.b = np.random.normal(0, 0.2, (output_shape))

        self.next_layer = None
        self.last_layer = None

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.x = np.zeros((input_shape))
        self.A = np.zeros((output_shape))
        self.B = np.zeros((output_shape))
        self.y = np.zeros((output_shape))

        self.dl_dw = np.zeros((output_shape, input_shape))
        self.dl_db = np.zeros((output_shape))
        self.dl_dx = np.zeros((input_shape))

    def forward(self, x):
        self.x = x
        self.A = self.x @ self.W.T
        self.B = self.A + self.b
        for i in range(len(self.B)):
            self.y[i] = self.a(self.B[i])
        return self.y

    def backward(self, dln_dx):
        for j in range(self.output_shape):
            for i in range(self.input_shape):
                self.dl_dw[j][i] = self.x[i] * self.a.d(self.B[j]) * dln_dx[j]
            self.dl_db[j] = self.a.d(self.B[j]) * dln_dx[j]

        for i in range(self.input_shape):
            self.dl_dx[i] = 0
            w = self.W[:, i]
            for j in range(self.output_shape):
                self.dl_dx[i] += self.a.d(self.B[j]) * w[j] * dln_dx[j]

        return self.dl_dx

    def __call__(self, x):
        outs = x @ self.W.T + self.b
        for ind in range(len(outs)):
            outs[ind] = self.a(outs[ind])
        return outs


if __name__ == "__main__":
    layer = Layer(2, 2)
    print(layer.forward(np.ones(2)))
    print(layer.backward(np.ones(2)))
