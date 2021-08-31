import numpy as np
from activations import relu, DERIV_MAP


class Layer:
    def __init__(self,
                 input_shape,
                 output_shape,
                 activation=relu):
        self.W = np.ones((input_shape, output_shape))
        self.b = np.ones((output_shape))
        self.a = activation
        self.da = DERIV_MAP[self.a]
        self.next_layer = None
        self.last_layer = None

        self.x = np.ones((input_shape))
        self.A = np.ones((input_shape))
        self.B = np.ones((input_shape))
        self.y = np.zeros((output_shape))

        self.dl_dw = np.ones((input_shape, output_shape))
        self.dl_db = np.ones((output_shape))
        self.dl_dx = np.ones((input_shape))

    def forward(self, x):
        self.x = x
        self.A = self.W @ self.x
        self.B = self.A + self.b
        for i in range(len(self.B)):
            self.y[i] = self.a(self.B[i])
        return self.y

    def backward(self):
        # todo: Recombine all these terms within two for loops at most

        dln_dx = self.next_layer.dl_dx
        for i in range(len(self.B)):
            for j in range(len(self.B)):
                self.dl_dw[i][j] = self.x[i] * self.da(self.B[j]) * dln_dx[j]

        for j in range(len(self.B)):
            self.dl_db[j] = self.da(self.B[j]) * dln_dx[j]

        for i in range(len(self.B)):
            self.dl_dx[i] = 0
            w = self.W[i]
            for j in range(len(self.B)):
                self.dl_dx[i] += self.da(self.B[j]) * w[j] * dln_dx[j]

    def __call__(self, x):
        outs = self.W @ x + self.b
        for ind in range(len(outs)):
            outs[ind] = self.a(outs[ind])
        return outs


if __name__ == "__main__":
    layer = Layer(10, 10)
    print(layer.forward(np.ones(10)))
    print(layer.backward())
