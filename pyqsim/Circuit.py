from typing import List

from .compute.swap import obtain_swaps
from .gates import Gate, SWAP, H, X, CX
from . import StateVectorRegister


class Circuit:
    register: StateVectorRegister
    gates: List

    def __init__(self, register: StateVectorRegister):
        self.register = register
        self.gates = []

    def h(self, *qubits: int):
        """Apply the Hadamard gate to the register."""
        self.add_gate(H(*qubits))

    def x(self, *qubits: int):
        """Apply the NOT gate to the register."""
        self.add_gate(X(*qubits))

    def cx(self, *qubits: int):
        """Apply the NOT gate to the register."""
        self.add_gate(CX(*qubits))

    def add_gate(self, gate: Gate):
        if gate.gate_size == 1:
            self.gates.append(gate)
        else:
            desired_state = gate.controls + gate.targets
            desired_state += [qubit for qubit in self.register.qubits if qubit not in desired_state]
            swaps = obtain_swaps(tuple(self.register.qubits), tuple(desired_state))
            reverse_swaps = swaps[::-1]
            self.gates += [SWAP(qubit_0, qubit_1, self.register.num_qubits) for qubit_0, qubit_1 in swaps]
            self.gates.append(gate)
            self.gates += [SWAP(qubit_0, qubit_1, self.register.num_qubits) for qubit_0, qubit_1 in reverse_swaps]

    def run(self):
        for gate in self.gates:
            self.register.apply_gate(gate)
        return self.register
