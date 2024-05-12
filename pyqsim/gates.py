from functools import lru_cache
from typing import List, Type

import numpy as np


class Gate(object):
    matrix: np.ndarray
    gate_size: int
    controls: List[int] = []
    targets: List[int] = []
    qubits: List[int] = []

    def __init__(self, *qubits: int) -> None:
        self.gate_size = np.log2(self.matrix.shape[0]).astype(int)
        self.assign_qubits(qubits)

    def assign_qubits(self, qubits):
        self.targets = list([qubits[-1]])
        self.controls = list(qubits[:-1])
        self.qubits = list(qubits)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        cls.matrix = matrix
        return cls


class H(Gate):
    matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


class X(Gate):
    matrix = np.array([[0, 1], [1, 0]], dtype=bool)


class CX(Gate):
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=bool)


class SWAP(Gate):
    def __init__(self, qubit_0: int, qubit_1: int, register_size: int):
        self.matrix = get_swap_matrix(qubit_0, qubit_1, register_size)
        super().__init__(qubit_0, qubit_1)

    def assign_qubits(self, qubits):
        self.targets = list(range(self.matrix.shape[0]))
        self.qubits = list(range(self.matrix.shape[0]))


@lru_cache(maxsize=None)
def get_swap_matrix(qubit_0: int, qubit_1: int, register_size: int) -> np.ndarray:
    """Compute the swap gate for a list of qubits."""
    mask_q0 = 1 << qubit_0
    mask_q1 = 1 << qubit_1
    matrix_size = 1 << register_size
    swap_matrix = np.eye(matrix_size, dtype=bool)
    for q0_index in range(matrix_size):
        if q0_index & mask_q0 and not (q0_index & mask_q1):
            q1_index = q0_index - mask_q0 + mask_q1
            swap_matrix[q0_index, q0_index] = 0
            swap_matrix[q1_index, q1_index] = 0
            swap_matrix[q0_index, q1_index] = 1
            swap_matrix[q1_index, q0_index] = 1
    return swap_matrix
