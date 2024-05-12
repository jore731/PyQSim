import unittest

import numpy as np

from pyqsim import Circuit, StateVectorRegister


class TestFull(unittest.TestCase):
    def test_full_circuit(self):
        circuit = Circuit(register=StateVectorRegister(4))

        circuit.h(0)
        circuit.h(1)
        circuit.h(3)
        circuit.cx(0, 1)
        circuit.h(1)
        circuit.cx(0, 2)
        # circuit.register.measure(0)
        circuit.x(2)
        circuit.cx(2, 3)
        circuit.run()

        expected = np.zeros((16, 1), dtype=complex)
        expected[4] = expected[5] = expected[12] = expected[13] = 1 / 2

        np.testing.assert_almost_equal(circuit.register.state, expected)

if __name__ == '__main__':
    unittest.main()
