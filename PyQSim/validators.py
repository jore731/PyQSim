import numpy as np


def registry_creation(
    num_qubits: int, data: np.ndarray = None, initial_state: int = None
):
    if data and initial_state:
        raise ValueError("You cannot provide both data and initial_state")
    elif data:
        if data.shape != (2**num_qubits, 1):
            raise ValueError("The data shape does not match the number of qubits")
        if not np.allclose(np.linalg.norm(data), 1):
            raise ValueError("The data must be normalized")
        if data.dtype is not complex:
            raise ValueError("The data must be a complex array")
    elif initial_state:
        if isinstance(initial_state, str):
            try:
                initial_state = int(initial_state, 2)
            except ValueError:
                raise ValueError(
                    "The initial state must be a binary string or an integer"
                )
        if initial_state >= 2**num_qubits or initial_state < 0:
            raise ValueError("The initial state is out of bounds")
