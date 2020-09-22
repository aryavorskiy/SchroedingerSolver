import csv

import numpy as np

from LinearAlgebraModel.Model.Grid import Grid
from LinearAlgebraModel.Model.Operators import AbstractOperator


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

    def __init__(self, **kwargs):
        if 'hamiltonian' in kwargs and 'grid' in kwargs:
            ham, grid = kwargs['hamiltonian'], kwargs['grid']
            mat = np.array(ham.get_matrix(grid), dtype='complex')
            eig = np.linalg.eig(mat)
            self.values, self.solution_matrix = eig
            self.grid = grid
            self.states = [WaveFunction(self.grid, line) for line in self.solution_matrix.transpose()]
        if 'filename' in kwargs:
            if not kwargs['filename'].endswith('.csv'):
                raise ValueError('Cannot load solution from non-csv file')
            self.load(kwargs['filename'])

    def dump(self, filename: str):
        """
        Dump solution to a CSV file

        :param filename: File to dump solution to
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        writer = csv.writer(open(filename, 'w'))
        writer.writerow([b[0] for b in self.grid.bounds])
        writer.writerow([b[1] for b in self.grid.bounds])
        writer.writerow(self.grid.sizes)
        writer.writerow(self.values)
        for line in self.solution_matrix:
            writer.writerow(line)

    def load(self, filename):
        """
        Loads solution from provided CSV file

        :param filename: File to load solution from
        """
        solution_data = list(csv.reader(open(filename)))
        l_bounds = [float(a) for a in solution_data[0]]
        u_bounds = [float(a) for a in solution_data[1]]
        sizes = [int(a) for a in solution_data[2]]
        values = [complex(a) for a in solution_data[3]]
        self.grid = Grid(list(zip(l_bounds, u_bounds)), sizes)
        self.values = values
        solution_data = solution_data[4:]
        self.solution_matrix = np.array(solution_data, dtype='complex')
        self.states = [WaveFunction(self.grid, line) for line in self.solution_matrix.transpose()]

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
