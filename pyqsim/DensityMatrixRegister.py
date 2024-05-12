from typing import Union

import numpy as np

from pyqsim.exceptions import RegistrySizeError
from pyqsim.gates import Gate


class DensityMatrixRegister:
    num_qubits: int
    density_matrix: np.ndarray

    # @validate(validators.registry_creation, is_classmethod=True)
    def __init__(
            self,
            num_qubits: int,
            density_matrix: np.ndarray = None,
            initial_state: Union[int, str] = None,
    ):
        self.num_qubits = num_qubits
        self.qubits = tuple(range(num_qubits))

        if density_matrix is not None:
            self.density_matrix = density_matrix
        else:
            if initial_state:
                if isinstance(initial_state, str):
                    initial_state = int(initial_state, 2)
            else:
                initial_state = 0

            self.density_matrix = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
            self.density_matrix[initial_state, initial_state] = 1

    def apply_gate(self, gate: Gate):
        if gate.gate_size > self.num_qubits:
            raise RegistrySizeError("Registry is too small to apply selected gate")
        matrix = 1
        for qubit in self.qubits:
            if qubit not in gate.qubits:
                matrix = np.kron(np.eye(2), matrix)
            elif qubit == gate.qubits[0]:
                matrix = np.kron(gate.matrix, matrix)
        self.density_matrix = matrix @ self.density_matrix @ matrix.conj().T

    def measure(self, qubit):
        series = np.repeat([0, 1], 2 ** qubit)
        occurrence = int(2 ** self.num_qubits / 2 ** (qubit + 1))
        ones = np.tile(series, occurrence).astype(bool)
        zeros = (np.ones(len(ones)) - ones).astype(bool)

        p = np.sum(np.absolute(ones * self.density_matrix.T) ** 2)

        if np.random.rand() < p:
            self.density_matrix = (self.density_matrix.T * zeros / np.sqrt(p)).T
            return 1
        else:
            self.density_matrix = (self.density_matrix.T * ones / np.sqrt(1 - p)).T
            return 0
