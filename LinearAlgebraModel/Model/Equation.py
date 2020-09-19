import numpy as np
from LinearAlgebraModel.Model.Operators import AbstractOperator
from LinearAlgebraModel.Model.Grid import Grid


class WaveFunction:
    def __init__(self, grid, values):
        if len(grid) != len(values):
            raise ValueError('Value count does not correspond to grid size')
        self.grid = grid
        norm = sum(x * np.conj(x) for x in values)
        self.values = np.array([x / norm for x in values])

    def operator_value(self, operator: AbstractOperator):
        """
        Calculate the mean operator value for this wave function

        :param operator: Operator object
        :return: Value
        """
        return np.linalg.multi_dot(
            (self.values.transpose(), operator.get_matrix(self.grid), self.values))

    def operator_error(self, operator: AbstractOperator):
        """
        Calculate the operator value for this wave function

        :param operator: Operator object
        :return:
        """
        val = self.operator_value(operator)
        op = val * np.eye(len(self.grid)) - operator.get_matrix(self.grid)
        return np.linalg.multi_dot((self.values.transpose(), op, op, self.values))


class SchroedingerSolution:
    """
    A structure that contains solutions of a Schroedinger equation. States can be obtained by energy
    """
    def __init__(self, hamiltonian: AbstractOperator, grid: Grid):
        mat = np.array(hamiltonian.get_matrix(grid), dtype='complex')
        eig = np.linalg.eig(mat)
        self.values = eig[0]
        self.states = [WaveFunction(grid, line) for line in eig[1].transpose()]

    def __getitem__(self, energy):
        """
        Obtain state with energy nearest to given

        :return: State
        """
        nearest_idx = min(list(range(len(self.values))), key=lambda i: abs(energy - self.values[i]))
        return self.states[nearest_idx]

    def __iter__(self):
        """
        Iterate over all states

        :return: Iterator
        """
        return iter(self.states)
