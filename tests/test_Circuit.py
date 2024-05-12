import unittest
from unittest.mock import Mock, MagicMock

from pyqsim.gates import H, CX, X, SWAP
from pyqsim import Circuit


def get_mock_register(size):
    """Return a Mock instance that implements the methods:

    qubits = return range(size)
    num_qubits = size
    """
    mock_register = MagicMock()
    mock_register.qubits = list(range(size))
    mock_register.num_qubits = size
    mock_register.apply_gate = MagicMock()
    return mock_register


class TestCircuit(unittest.TestCase):
    def setUp(self):
        self.mock_register = get_mock_register(2)
        self.circuit = Circuit(self.mock_register)

    def test_init(self):
        self.assertEqual(self.circuit.register, self.mock_register)

    def test_h(self):
        self.circuit.h(1)
        self.assertIsInstance(self.circuit.gates[0], H)

    def test_x(self):
        self.circuit.x(1)
        self.assertIsInstance(self.circuit.gates[0], X)

    def test_cx(self):
        self.circuit.cx(0, 1)
        self.assertIsInstance(self.circuit.gates[0], CX)

    def test_cx_swapped(self):
        self.circuit.cx(1, 0)
        self.assertIsInstance(self.circuit.gates[0], SWAP)
        self.assertIsInstance(self.circuit.gates[1], CX)
        self.assertIsInstance(self.circuit.gates[2], SWAP)

    def test_run_plus(self):
        self.circuit.h(0)
        self.circuit.cx(0, 1)

        expected_gates = [H, CX]
        self.circuit.run()
        for call, gate in zip(self.mock_register.apply_gate.call_args_list, expected_gates):
            self.assertIsInstance(call.args[0], gate)

    def test_run_minus(self):
        self.circuit.h(1)
        self.circuit.cx(1, 0)

        expected_gates = [H,SWAP, CX, SWAP]
        self.circuit.run()
        for call, gate in zip(self.mock_register.apply_gate.call_args_list, expected_gates):
            self.assertIsInstance(call.args[0], gate)


if __name__ == "__main__":
    unittest.main()
