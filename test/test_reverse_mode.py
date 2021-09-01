import unittest
from src.reverse_mode import Var as VarRM
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
        x = VarRM(2, name='x')
        y = VarRM(1, name='y')
        output = (x + y) * x + (x * x * x)
        do_dy_rev, do_dx_rev = output.compute_grad()[-2:]
        self.assertEqual(
            (do_dx_rev, do_dy_rev),
            (do_dx_for, do_dy_for),
            torch_grads
        )


if __name__ == '__main__':
    unittest.main()
