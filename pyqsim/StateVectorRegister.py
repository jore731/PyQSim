from typing import Union, List

import numpy as np

from . import validators
from .exceptions import RegistrySizeError
from .gates import Gate
from .utils import validate


class StateVectorRegister:
    """Quantum register that gets initialized with a certain number of qubits."""

    num_qubits: int
    qubits: List[int]

    @validate(validators.registry_creation, is_classmethod=True)
    def __init__(
            self,
            num_qubits: int,
            data: np.ndarray = None,
            initial_state: Union[int, str] = None,
    ):
        self.num_qubits = num_qubits
        self.qubits = list(range(num_qubits))

        if data is not None:
            self.state = data
        else:
            if initial_state:
                if isinstance(initial_state, str):
                    initial_state = int(initial_state, 2)
            else:
                initial_state = 0

            self.state = np.zeros((2 ** num_qubits, 1), dtype=complex)
            self.state[initial_state] = 1

    @property
    def ket(self):
        """Return the state vector of the register as a ket."""
        return self.state

    @property
    def bra(self):
        """Return the state vector of the register as a bra."""
        return self.state.T.conj()

    @property
    def density_matrix(self):
        """Return the density matrix of the register."""
        return self.ket @ self.bra

    def apply_gate(self, gate: Gate):
        """
        Apply a single gate to the register
        Time complexity:
          O(2*2^num_qubits)
          ω(2*2^num_qubits)

        Space complexity:
          O(2^num_qubits)
          ω(2^num_qubits)
        """
        if gate.gate_size > self.num_qubits:
            raise RegistrySizeError("Registry is too small to apply selected gate")
        matrix = 1
        for qubit in self.qubits:
            if qubit not in gate.qubits:
                matrix = np.kron(np.eye(2), matrix)
            elif qubit == gate.qubits[0]:
                matrix = np.kron(gate.matrix, matrix)
        self.state = matrix @ self.ket

    def measure(self, qubit):
        """
        Measure a qubit within the register. When measured, the state of the register collapses.

        Time complexity:
          O(2^n)
          ω(2^n)

        Space complexity:
          O(2^n)
          ω(2^n)
        """
        series = np.repeat([0, 1], 2 ** qubit)
        occurrence = int(2 ** self.num_qubits / 2 ** (qubit + 1))
        ones = np.tile(series, occurrence).astype(bool)
        zeros = (np.ones(len(ones)) - ones).astype(bool)

        p = np.sum(np.absolute(ones * self.state.T) ** 2)

        if np.random.rand() < p:
            self.state = (self.state.T * zeros / np.sqrt(p)).T
            return 1
        else:
            self.state = (self.state.T * ones / np.sqrt(1 - p)).T
            return 0

    @property
    def bloch_sphere(self):
        """Calculate the bloch spehere"""
        if self.num_qubits > 1:
            raise NotImplementedError()

        for index, state in enumerate(self.state):
            if state:
                arg = np.angle(state)
                canonical = np.exp(-arg * 1j) * self.state
                canonical[index] = canonical[index].real
                phi = np.angle(canonical[1])[0]
                theta = 2 * np.arccos(canonical[0].real)[0]
                return np.array([phi, theta])

    # ---------------------- Support Methods ----------------------------------
    def visualize(self, only_ones=False):
        """
        Visualize the state vector in a table with the corresponding states and its value.
        """
        states = [f"|{i:0{self.num_qubits}b}>" for i in range(2 ** self.num_qubits)]
        values = [
            f"{np.round(self.state[i].real ** 2, 2)}"
            for i in range(2 ** self.num_qubits)
        ]
        print(f"{'State':<10}{'Value':<10}")
        for state, value in zip(states, values):
            if not only_ones or value == "1.0":
                print(f"{state:<10}{value:<10}")
