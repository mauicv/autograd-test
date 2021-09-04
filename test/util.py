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
