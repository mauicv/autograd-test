# import numpy as np

"""Activations"""


class LeakyRelu:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return x * (x > 0) + self.a * x * (x <= 0)

    def d(self, x):
        return 1 * (x > 0) + self.a * (x <= 0)


class Relu(LeakyRelu):
    def __init__(self):
        super().__init__(0)


class Linear:
    def __init__(self, a=1):
        self.a = a

    def __call__(self, x):
        return x * self.a

    def d(self, x):
        return self.a

#
# class Tanh:
#     def __init__(self):
#         pass
#
#     def __call__(self, x):
#         return np.tanh(x)
#
#     def d(x):
#         return 1 - np.tanh(x)**2


"""Losses"""


class Mse:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return ((x - y)**2).mean()

    def d(self, x, y):
        return 2*(x**2 - y)/len(y)
