import qutip as qt
import numpy as np
from collections import defaultdict


class System:
    """ Any quantum system that has a state (vector or density matrix) and some number of operators. """
    def __init__(self, state, **operators):
        self.state = state
        for key, val in operators.items():
            setattr(self, key, val)

    def __getattr__(self, name):
        """ Tell static code analysis tools the attributes are set dynamically. """
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def operator_names(self):
        """ Returns the names of all operators currently held in this system. """
        return self.__dict__.keys() - {'state'}

    @property
    def dims(self):
        """ Returns the first set of tensor dimensions of the state. (not matrix dimensions) """
        return self.state.dims[0]

    def get_operator_list(self, name):
        """ Returns the operator(s) stored under `name` as a list even if there is only one. """
        operator_list = getattr(self, name, None)
        return operator_list if isinstance(operator_list, list) else [operator_list]

    @property
    def subsystems(self):
        """ The number of subsystem operators currently held in this system. """
        return len(self.get_operator_list(self.operator_names.pop()))

    def __mul__(self, other):
        """ Combine the systems into the same hilbert space. """
        if isinstance(other, System):
            # get a list of all operator names in both systems
            names = self.operator_names | other.operator_names
            # create an operators dict to hold lists of operators
            operators = defaultdict(list)
            for name in names:
                # expand this system's operators by the dimensions of the other
                for op in self.get_operator_list(name):
                    operators[name].append(None if op is None else qt.tensor(op, qt.qeye(other.dims)))
                # maintain alignment of operator lists
                operators[name] += [None] * (self.subsystems - len(operators[name]))
                # expand other systems operators by the dimensions of this one
                for op in other.get_operator_list(name):
                    operators[name].append(None if op is None else qt.tensor(qt.qeye(self.dims), op))
            # return the multipart system, where the new state is just the tensor product of the two subsystems
            return System(qt.tensor(self.state, other.state), **operators)

    @property
    def as_list(self):
        """ Break down this multipart system with lists of subsystem operators into a list of subsystems. """
        dicts = [{} for _ in range(self.subsystems)]
        for name in self.operator_names:
            for i, operator in enumerate(self.get_operator_list(name)):
                dicts[i][name] = operator
        return [System(self.state, **dictionary) for dictionary in dicts]


def expectation(operator, states):
    """ Calculate the expectation value of an operator for a series of states. """
    return np.array([np.trace(operator @ state).real for state in states])


def make_lindblad_equation(hamiltonian, collapse_operators):
    """
    Creates a closure to compute the right-hand side of the Lindblad master equation
    for a given Hamiltonian and set of collapse operators.

    The Lindblad master equation describes the time evolution of a quantum system's
    density matrix in a non-unitary way due to interactions with an environment.

    Parameters:
    - hamiltonian (function): A function that takes time 't' as input and returns the
      Hamiltonian matrix H(t) at that time. The Hamiltonian should already be scaled
      appropriately by physical constants (e.g., reduced Planck constant).
    - collapse_operators (list of arrays): A list of operator matrices that represent
      the system's interaction with its environment, contributing to its decoherence
      and dissipation.

    Returns:
    - lindblad_rhs (function): A function that takes time 't' and the state vector
      'rho_vec' (flattened density matrix) as inputs and returns the time derivative
      of 'rho_vec' according to the Lindblad equation. The returned function is a closure
      that captures the 'hamiltonian' and 'collapse_operators' from its defining scope.

    Example usage:
    ```
    # Define the Hamiltonian as a function of time
    def my_hamiltonian(t):
        return np.array([[0, np.exp(-1j * t)], [np.exp(1j * t), 0]])

    # Define collapse operators
    collapse_ops = [np.array([[1, 0], [0, 0]])]

    # Create the Lindblad equation solver
    lindblad_solver = make_lindblad_equation(my_hamiltonian, collapse_ops)

    # Use the solver at a specific time and state
    t = 0.0
    rho_vec = np.array([1, 0, 0, 1]).ravel()  # Example initial state vector
    drho_dt = lindblad_solver(t, rho_vec)
    ```
    """
    def lindblad_rhs(t, rho_vec):
        """ Compute the right-hand side of the Lindblad master equation. """
        H = hamiltonian(t)
        rho = rho_vec.reshape(H.shape)
        # result = (-1j / hbar) * (H @ rho - rho @ H)   # if hamiltonian was not divided by hbar before integration
        result = -1j * (H @ rho - rho @ H)
        for operator in collapse_operators:
            result += operator @ rho @ operator.conj().T
            result -= 0.5 * (operator.conj().T @ operator @ rho + rho @ operator.conj().T @ operator)
        return result.ravel()

    return lindblad_rhs
