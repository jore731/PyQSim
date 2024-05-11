from loguru import logger
import numpy as np
from typing import Union, Set

from . import gates, validators
from .compute.swap import compute_swap_gate, obtain_swaps, compute_swap_block
from .utils import as_list, validate


class Register:
    """Quantum register that gets initialized with a certain number of qubits."""

    num_qubits: int
    qubits: Set[int]

    @validate(validators.registry_creation, is_classmethod=True)
    def __init__(
        self,
        num_qubits: int,
        data: np.ndarray = None,
        initial_state: Union[int, str] = None,
    ):
        self.num_qubits = num_qubits
        self.qubits = set(range(num_qubits))
        self.current_qubits = self.qubits.copy()
        self.applied_swaps = tuple()

        if data:
            self.state = data
        else:
            if initial_state:
                if isinstance(initial_state, str):
                    initial_state = int(initial_state, 2)
            else:
                initial_state = 0

            self.state = np.zeros((2**num_qubits, 1), dtype=complex)
            self.state[initial_state] = 1

        self.pending_gates = {}

    @property
    def desired_qubits(self):
        desired_qubits = []
        for _, qubits in self.pending_gates.values():
            desired_qubits += qubits
        return desired_qubits

    @property
    def expected_arrangement(self):
        missing_qubits = set(self.desired_qubits) ^ set(self.qubits)
        return self.desired_qubits + list(missing_qubits)

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

    def queue_gate(self, gate, target_qubits):
        unavailable_qubits = set(self.desired_qubits)
        available_qubits = set(target_qubits) ^ set(unavailable_qubits)

        if not set(target_qubits).issubset(available_qubits):
            self.apply_now()

        self.pending_gates[target_qubits[0]] = (gate, target_qubits)

    def apply_now(self):
        logger.info(f"Applying {len(self.pending_gates)} gates")
        logger.debug(f"Applying {self.pending_gates}")
        self._prepare_to_apply()
        self._apply_gates()
        self._reverse_swaps()
        self.pending_gates = {}

    def apply_swaps(self, swaps):
        logger.info(f"Applying {len(swaps)} swaps")
        logger.debug(swaps)
        swap_block = compute_swap_block(swaps, self.num_qubits)
        self.state = swap_block @ self.state
        self.current_qubits = self.expected_arrangement
        for swap in swaps:
            if self.applied_swaps and set(swap) == set(self.applied_swaps[-1]):
                self.applied_swaps = self.applied_swaps[:-1]
            else:
                self.applied_swaps += ((swap),)
            logger.debug(f"Applied swaps: {self.applied_swaps}")

    def _prepare_to_apply(self):
        """Rearrange the register to apply the gates."""
        logger.debug(
            f"Transforming {tuple(self.current_qubits)} into {tuple(self.expected_arrangement)}"
        )
        swaps = obtain_swaps(
            tuple(self.current_qubits), tuple(self.expected_arrangement)
        )
        logger.debug(f"Preparation swaps: {swaps}")
        self.apply_swaps(swaps)

    def _apply_gates(self):
        """Apply a gate to the register."""
        buffer = 1
        skip_next = []
        for qubit in self.current_qubits:
            if skip_next:
                skip_next.pop(0)
                continue
            if qubit in self.pending_gates:
                gate, target_qubits = self.pending_gates.pop(qubit)
                buffer = np.kron(gate.matrix, buffer)
                skip_next = target_qubits[:-1]
            else:
                buffer = np.kron(np.eye(2), buffer)
        logger.debug("Buffer")
        logger.debug(buffer)
        logger.debug("state")
        logger.debug(self.state)
        self.state = buffer @ self.state

    def _reverse_swaps(self):
        swaps = tuple(reversed(self.applied_swaps))
        self.apply_swaps(swaps)

    def measure(self, qubit):
        """
        Measure the register.

        p(n)= O(2**{n})
        p(n)= W(2**{n})
        p(n)= θ(2**{n})
        """
        series = np.repeat([0, 1], 2**qubit)
        ocurrence = int(2**self.num_qubits / 2 ** (qubit + 1))
        ones = np.tile(series, ocurrence)
        zeros = np.ones(len(ones)) - ones

        p = np.sum(np.absolute(ones * self.state.T) ** 2)

        if np.random.rand() < p:
            for index in zeros:
                self.state[index, 0] = 0
                self.state /= np.sqrt(p)
            return 1
        else:
            for index in ones:
                self.state[index, 0] = 0
                self.state /= np.sqrt(1 - p)
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
                phi = np.angle(canonical[1])
                theta = 2 * np.arccos(canonical[0].real)
                return phi, theta

    ######################## Support Methods ##################################

    def visualize(self, only_ones=False):
        """
        Visualize the state vector in a table with the corresponding states and its value.
        """
        states = [f"|{i:0{self.num_qubits}b}>" for i in range(2**self.num_qubits)]
        values = [
            f"{np.round(float(self.state[i][0]**2),2)}"
            for i in range(2**self.num_qubits)
        ]
        print(f"{'State':<10}{'Value':<10}")
        for state, value in zip(states, values):
            if not only_ones or value == "1.0":
                print(f"{state:<10}{value:<10}")
