
def relu(x):
    return x * (x > 0)


def relu_der(x):
    return 1 * (x > 0)


DERIV_MAP = {
    relu: relu_der
}
