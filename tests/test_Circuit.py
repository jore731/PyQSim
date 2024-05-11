import unittest
from unittest.mock import Mock, call
from PyQSim import Circuit


class TestCircuit(unittest.TestCase):
    def setUp(self):
        self.mock_register = Mock()
        self.circuit = Circuit(self.mock_register)

    def test_init(self):
        self.assertEqual(self.circuit.registry, self.mock_register)
        self.assertEqual(self.circuit.gates, a)

    def test_h(self):
        self.circuit.h(1)
        self.assertEqual(self.circuit.gates, [call.H(1)])

    def test_x(self):
        self.circuit.x(1)
        self.assertEqual(self.circuit.gates, [call.X(1)])

    def test_cx(self):
        self.circuit.cx(1, 2)
        self.assertEqual(self.circuit.gates, [call.CX(1, 2)])


if __name__ == "__main__":
    unittest.main()
