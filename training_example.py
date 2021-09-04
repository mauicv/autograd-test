from src.model import Model
from src.layer import Layer
from src.functions import Linear, LeakyRelu
from test.util import fn_data_gen
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm


if __name__ == '__main__':
    layer_dims = [2, 30, 30, 1]
    losses = []
    batch_size = 32
    training_steps = 20000
    count = 0

    # Model:
    model = Model()
    layers = [Layer(d1, d2, activation=LeakyRelu(0.1)) for d1, d2
              in zip(layer_dims[:-1], layer_dims[1:-1])]
    last_layer = Layer(layer_dims[-2], layer_dims[-1],
                       activation=Linear())
    layers.append(last_layer)
    model.add_layers(layers)
    model.compile()

    with tqdm(total=training_steps) as pbar:
        for X, Y in fn_data_gen(training_steps):
            pbar.update(1)
            if (count % batch_size) == 0:
                grad_avg = [(np.zeros_like(layer.W), np.zeros_like(layer.b))
                            for layer in model.layers[::-1]]
            count += 1
            loss, grads = model.forward(X, Y)
            for (avg_w_grad, avg_b_grad), (weight_grads, bias_grads) \
                    in zip(grad_avg, grads):
                avg_w_grad += weight_grads
                avg_b_grad += bias_grads
            losses.append(loss)

            if (count % batch_size) == 0 and count > 0:
                for layer, (weight_grads, bias_grads) \
                        in zip(model.layers[::-1], grad_avg):
                    coef = -0.075 / batch_size
                    layer.W += coef * weight_grads
                    layer.b += coef * bias_grads

    losses = np.convolve(
        np.array(losses),
        np.ones(10*batch_size)/(10*batch_size),
        mode='valid')

    for X, Y in fn_data_gen(5):
        Y_pred = model(X)
        print('input:', X.round(3),
              'output:', Y_pred.round(3)[0],
              'true:', Y.round(3)[0],
              'error:', (Y_pred-Y).round(3)[0])

    plt.plot(losses[1:])
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    X = np.arange(-1, 1, 0.025)
    Y = np.arange(-1, 1, 0.025)
    X, Y = np.meshgrid(X, Y)
    Z = (X**2 + Y**2)/2
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    Z = np.array([model(np.array([x, y])) for x, y
                  in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    plt.show()
