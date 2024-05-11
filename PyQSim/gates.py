from typing import List
import numpy as np


class Gate:
    matrix: np.ndarray
    affected_qubits: int
    controls: List[int]
    target: int

    def __init__(self, *qubits) -> None:
        self.affected_qubits = np.log2(self.matrix.shape[0]).astype(int)
        self.target = qubits[-1]
        self.controls = qubits[:-1]


class H(Gate):
    matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


class X(Gate):
    matrix = np.array([[0, 1], [1, 0]])


class CX(Gate):
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
