from loguru import logger
import numpy as np
from functools import lru_cache

from typing import List, Tuple


@lru_cache(maxsize=None)
def compute_swap_gate(q0: int, q1: int, num_qubits: int) -> np.ndarray:
    """Compute the swap gate for a list of qubits."""
    mask_q0 = 1 << q0
    mask_q1 = 1 << q1
    matrix_size = 1 << num_qubits
    swap = np.eye(matrix_size, dtype=bool)
    for q0_index in range(matrix_size):
        if q0_index & mask_q0 and not (q0_index & mask_q1):
            q1_index = q0_index - mask_q0 + mask_q1
            swap[q0_index, q0_index] = 0
            swap[q1_index, q1_index] = 0
            swap[q0_index, q1_index] = 1
            swap[q1_index, q0_index] = 1
    return swap


@lru_cache(maxsize=None)
def obtain_swaps(current_state: Tuple, desired_state: Tuple):
    current_state = np.array(current_state, dtype=int)
    desired_state = np.array(desired_state, dtype=int)
    swaps = tuple()
    logger.debug(f"Desired state: {desired_state}")
    while all(current_state != desired_state):
        for index, qubit in enumerate(current_state):
            logger.debug(f"Current state: {current_state}")
            if qubit == desired_state[index]:
                continue
            else:
                target = np.where(current_state == desired_state[index])
                swaps += ((index, target[0][0]),)
                current_state[target] = current_state[index]
                current_state[index] = desired_state[index]
    return swaps


@lru_cache(maxsize=None)
def compute_swap_block(swaps, registry_size):
    swap_block = np.eye(1 << registry_size, dtype=bool)
    for swap in swaps:
        swap_block = compute_swap_gate(*swap, registry_size) @ swap_block
    return swap_block
