import unittest

import numpy as np

from pyqsim import StateVectorRegister, gates
from pyqsim.exceptions import RegistrySizeError


class TestRegisterCreation(unittest.TestCase):
    def test_create_register(self):
        register = StateVectorRegister(4)

        expected_state = np.zeros((16, 1), dtype=complex)
        expected_state[0, 0] = 1

        self.assertIsInstance(register, StateVectorRegister)
        self.assertEqual(register.num_qubits, 4)
        self.assertEqual(register.qubits, list(range(4)))
        np.testing.assert_allclose(register.state, expected_state)

    def test_create_register_from_data(self):
        known_state_vector = np.zeros((16, 1), dtype=complex)
        known_state_vector[0] = known_state_vector[5] = known_state_vector[7] = np.sqrt(1 / 3)

        register = StateVectorRegister(4, data=known_state_vector)

        self.assertIsInstance(register, StateVectorRegister)
        self.assertEqual(register.num_qubits, 4)
        self.assertEqual(register.qubits, list(range(4)))
        np.testing.assert_allclose(register.state, known_state_vector)

    def test_create_register_from_value(self):
        state = 4
        expected_state = np.zeros((16, 1), dtype=complex)
        expected_state[state] = 1

        register = StateVectorRegister(4, initial_state=state)
        self.assertIsInstance(register, StateVectorRegister)
        self.assertEqual(register.num_qubits, 4)
        self.assertEqual(register.qubits, list(range(4)))
        np.testing.assert_allclose(register.state, expected_state)


class TestRegisterFunctionalities(unittest.TestCase):
    def setUp(self):
        self.register = StateVectorRegister(4, initial_state=2)

    def test_ket(self):
        expected = np.zeros((16, 1), dtype=complex)
        expected[2] = 1
        np.testing.assert_allclose(self.register.ket, expected)

    def test_bra(self):
        expected = np.zeros((1, 16), dtype=complex)
        expected[0, 2] = 1
        np.testing.assert_allclose(self.register.bra, expected)

    def test_density_matrix(self):
        self.register = StateVectorRegister(2, initial_state=2)
        expected = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0]
            ], dtype=complex
        )
        np.testing.assert_allclose(self.register.density_matrix, expected)

    def test_apply_gate_same_size_as_register(self):
        self.register = StateVectorRegister(1)

        expected = np.zeros((2, 1), dtype=complex)
        expected[0] = expected[1] = np.sqrt(1 / 2)

        self.register.apply_gate(gates.H(0))

        np.testing.assert_allclose(self.register.state, expected)

    def test_apply_gate_smaller_than_register(self):
        expected = np.zeros((16, 1), dtype=complex)
        expected[2] = expected[6] = np.sqrt(1 / 2)
        self.register.apply_gate(gates.H(2))
        np.testing.assert_allclose(self.register.state, expected)

    def test_apply_multi_qubit_gate(self):
        self.register = StateVectorRegister(2)
        self.register.apply_gate(gates.H(0))
        self.register.apply_gate(gates.H(1))

        expected = np.zeros((4, 1), dtype=complex)
        expected[:] = 1 / 2
        self.register.apply_gate(gates.CX(1, 0))
        np.testing.assert_allclose(self.register.state, expected)

    def test_apply_gate_bigger_than_register(self):
        with self.assertRaises(RegistrySizeError):
            self.register = StateVectorRegister(1)
            self.register.apply_gate(gates.CX(2))

    def test_measure(self):
        self.register = StateVectorRegister(2)

        self.register.apply_gate(gates.H(0))
        self.register.apply_gate(gates.H(1))
        self.register.apply_gate(gates.CX(1, 0))

        c_0 = self.register.measure(0)

        if c_0:
            expected = np.zeros((4, 1), dtype=complex)
            expected[0] = expected[2] = np.sqrt(1 / 2)
        else:
            expected = np.zeros((4, 1), dtype=complex)
            expected[1] = expected[3] = np.sqrt(1 / 2)

        np.testing.assert_allclose(self.register.state, expected)

    def test_bloch_sphere(self):
        self.register = StateVectorRegister(1)

        self.register.apply_gate(gates.H(0))
        np.testing.assert_allclose(self.register.bloch_sphere, np.array([0, np.pi * 180 / 360]))


if __name__ == '__main__':
    unittest.main()
