import unittest

import numpy as np

from pyqsim import gates


class TestGates(unittest.TestCase):
    def test_X(self):
        x = gates.X(0)
        x_1 = gates.X(1)

        self.assertEqual(x.gate_size, 1)
        self.assertEqual(x.controls, [])
        self.assertEqual(x.targets, [0])
        self.assertEqual(x_1.targets, [1])
        self.assertEqual(x.matrix.size, 4)

    def test_CX(self):
        cx = gates.CX(0, 1)

        self.assertEqual(cx.gate_size, 2)
        self.assertEqual(cx.controls, [0])
        self.assertEqual(cx.targets, [1])
        self.assertEqual(cx.matrix.size, 16)

    def test_SWAP(self):
        register_size = 2
        swap_1023 = gates.SWAP(0, 1, register_size)

        self.assertEqual(swap_1023.gate_size, 2)
        self.assertEqual(swap_1023.controls, [])
        self.assertEqual(swap_1023.targets, list(range(1 << register_size)))
        self.assertEqual(swap_1023.matrix.tolist(),
                         [[True, False, False, False],
                          [False, False, True, False],
                          [False, True, False, False],
                          [False, False, False, True]])


if __name__ == '__main__':
    unittest.main()
