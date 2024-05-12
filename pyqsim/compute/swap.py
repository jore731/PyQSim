from loguru import logger
import numpy as np
from functools import lru_cache

from typing import List, Tuple

@lru_cache(maxsize=None)
def obtain_swaps(current_state: Tuple, desired_state: Tuple):
    current_state = np.array(current_state, dtype=int)
    desired_state = np.array(desired_state, dtype=int)
    swaps = tuple()
    logger.debug(f"Desired state: {desired_state}")
    while np.all(current_state != desired_state):
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
