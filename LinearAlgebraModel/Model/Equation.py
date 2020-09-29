import csv

import numpy as np

from LinearAlgebraModel.Model.Grid import Grid
from LinearAlgebraModel.Model.Operators import AbstractOperator


class WaveFunction:
    def __init__(self, grid, values):
        if len(grid) != len(values):
            raise ValueError('Value count does not correspond to grid size')
        self.grid = grid
        self.values = np.array(values, dtype=np.complex)
        self.values /= sum(x * np.conj(x) for x in self.values)

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

            print('Generating hamiltonian matrix...')
            mat = np.array(ham.get_matrix(grid), dtype=np.complex)

            print('Finding eigenvalues...')
            eig = np.linalg.eig(mat)

            print('Evaluating wave functions...')
            self.values = np.real_if_close(eig[0], tol=1E7)
            self.grid = grid
            self.states = [WaveFunction(self.grid, line) for line in eig[1].transpose()]
            print('Done.')
        if 'filename' in kwargs:
            print('Loading solution from file...')
            self.load(kwargs['filename'])
            print('Done.')

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
        for wf in self.states:
            writer.writerow(wf.values)

    def load(self, filename: str):
        """
        Loads solution from provided CSV file

        :param filename: File to load solution from
        """
        if not filename.endswith('.csv'):
            raise ValueError('Solution file must have CSV extension')
        solution_data = list(csv.reader(open(filename)))
        l_bounds = [float(a) for a in solution_data[0]]
        u_bounds = [float(a) for a in solution_data[1]]
        sizes = [int(a) for a in solution_data[2]]
        values = [complex(a) for a in solution_data[3]]
        self.grid = Grid(list(zip(l_bounds, u_bounds)), sizes)
        self.values = np.array(values)
        solution_data = solution_data[4:]
        self.states = [WaveFunction(self.grid, line) for line in solution_data]

    def __getitem__(self, args):
        """
        Obtain states with energy nearest to given

        :return: States tuple
        """
        if type(args) == tuple:
            energy, tolerance = args
        else:
            energy = args
            tolerance = 1

        nearest_nrg = min(list(self.values), key=lambda current_energy: abs(energy - current_energy))
        return tuple(self.states[i] for i in range(len(self.values)) if abs(self.values[i] - nearest_nrg) < tolerance)

    def __iter__(self):
        """
        Iterate over all states

        :return: Iterator
        """
        return iter(self.states)