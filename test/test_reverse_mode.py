import unittest
from src.reverse_mode import Var as VarRM, Tape
from src.forward_mode import Var as VarFD
import torch


class BackPropTests(unittest.TestCase):
    def test_correct_derivative(self):
        x = torch.tensor([2.], requires_grad=True)
        y = torch.tensor([1.], requires_grad=True)
        out_torch = (x + y) * x + (x * x * x)
        out_torch.backward(gradient=torch.tensor([1.]))
        torch_grads = (x.grad.tolist()[0], y.grad.tolist()[0])

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
        x, y = (VarRM(2), VarRM(1))
        tape = Tape()
        tape.watch([x, y])
        (x + y) * x + (x * x * x)
        tape.compute_grads()
        self.assertEqual(
            (x.df_dn, y.df_dn),
            (do_dx_for, do_dy_for),
            torch_grads
        )


if __name__ == '__main__':
    unittest.main()
