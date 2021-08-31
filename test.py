from src.reverse_mode import Var as VarRM
from src.forward_mode import Var as VarFD


if __name__ == "__main__":
    # forward_mode:
    x = VarFD(2, 1)
    y = VarFD(1, 0)
    out1 = (x + y) * x + (x * x * x)
    do_dx_for = out1.dot

    x = VarFD(2, 0)
    y = VarFD(1, 1)
    out1 = (x + y) * x + (x * x * x)
    do_dy_for = out1.dot

    # reverse_mode
    x = VarRM(2, name='x')
    y = VarRM(1, name='y')
    output = (x + y) * x + (x * x * x)
    do_dy_rev, do_dx_rev = output.compute_grad()[-2:]
    print((do_dx_rev, do_dy_rev))
    print((do_dx_for, do_dy_for))
    assert (do_dx_rev, do_dy_rev) == (do_dx_for, do_dy_for)
