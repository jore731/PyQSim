from typing import List

from . import Register, gates


class Circuit:
    registry: Register
    gates: List

    def __init__(self, registry: Register):
        self.registry = registry
        self.gates = []

    def h(self, *qubits: List[int]):
        """Apply the Hadamard gate to the register."""
        self.gates.append(gates.H(*qubits))

    def x(self, *qubits: List[int]):
        """Apply the NOT gate to the register."""
        self.gates.append(gates.X(*qubits))

    def cx(self, *qubits: List[int]):
        """Apply the NOT gate to the register."""
        self.gates.append(gates.CX(*qubits))
