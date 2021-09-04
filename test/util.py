import numpy as np


def print_layer_weights(layers1, layers2):
    for i, ((W, b), layer) in \
            enumerate(zip(layers1, layers2)):
        torch_dw = W.grad.detach().numpy()
        torch_db = b.grad.detach().numpy()

        print(f'============= layer {i} ==============')
        print('')
        print('------------- weights --------------')
        print('native grad:')
        print(layer.dl_dw.round(5))
        print('')
        print('torch grad:')
        print(torch_dw.round(5))
        print('')
        print('------------- bias -----------------')
        print('native grad:')
        print(layer.dl_db.round(5))
        print('torch grad:')
        print(torch_db.round(5))


def xor_data_gen(num=10):
    a = np.random.choice([1.0, 0.0], num)
    b = np.random.choice([1.0, 0.0], num)
    return zip(np.stack([a, b], axis=1), np.logical_xor(a, b)[:, None])


def fn_data_gen(num=10):
    x = np.random.uniform(low=0, high=1, size=num)
    y = np.random.uniform(low=0, high=1, size=num)
    fxy = (x**2 + y**2)/2
    return zip(np.stack([x, y], axis=1), fxy[:, None])


if __name__ == "__main__":
    for a, b in fn_data_gen():
        print(a, b)
