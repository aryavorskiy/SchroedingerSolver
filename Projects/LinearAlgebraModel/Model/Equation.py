import csv

import numpy as np

from General.Grid import Grid
from General.Utils import ProgressInformer
from Projects.LinearAlgebraModel.Model.BaseOperators import LinearOperator


class WaveFunction:
    """
    A class implementing a wave function.
    """

    def __init__(self, grid: Grid, values: list):
        """
        Creates a new WaveFunction object.

        :param grid: Grid object
        :param values: List with wave function values
        """
        if len(grid) != len(values):
            raise ValueError('Value count does not correspond to grid size')
        self.grid = grid
        self.values = np.array(list(complex(a) for a in values), dtype=np.complex)
        self.values /= sum(x * np.conj(x) for x in self.values) ** 0.5

    def operator_value_error(self, operator: LinearOperator):
        """
        Calculate the mean operator value and error for this wave function

        :param operator: Operator object
        :return: Operator value and error
        """
        val = np.linalg.multi_dot(
            (np.conj(self.values.transpose()), operator.mat, self.values))
        op = val * np.eye(len(self.grid)) - operator.mat
        column = np.conj(self.values) * np.dot(op, self.values)
        return complex(val), abs(np.dot(column.transpose(), column))

    def value_at(self, point):
        return self.values[self.grid.index(point)]


def naive_operator_value_error(wf: WaveFunction, op: LinearOperator):
    """
    Calculates eigenvalue of an operator on specified wave function.
    Omits values near to bounds or center.

    :param wf: WaveFunction object
    :param op: LinearOperator object
    :return: Operator value and error
    """
    op_values = np.dot(op.mat, wf.values)
    r = min(min(abs(k) for k in bounds) for bounds in wf.grid.bounds)
    avg_abs = 1 / len(wf.grid)
    values = []
    for pt in wf.grid.points_inside():
        if sum(a ** 2 for a in wf.grid.point_to_absolute(pt)) ** 0.5 > r / 4:
            wf_value = wf.value_at(pt)
            if abs(wf_value) > avg_abs / 10:
                values.append((op_values[wf.grid.index(pt)], wf_value))
    if len(values) == 0:
        return None, None
    avg = sum(pair[0] / pair[1] for pair in values) / len(values)
    err = (sum((abs((pair[0] / pair[1] - avg) * pair[1])) ** 2 for pair in values)) ** 0.5
    return avg, err


class SchrodingerSolution:
    """
    A structure that contains solutions of a Schrodinger equation.
    """

    def __init__(self, **kwargs):
        """
        Creates a new SchrodingerSolution instance.

        Specify hamiltonian or grid to solve directly, or filename to load form CSV table.

        :key hamiltonian: Hamiltonian operator object
        :key grid: Grid object
        :key filename: File to load solution from
        """
        print('Schrodinger equation initialization started')
        if 'hamiltonian' in kwargs and 'grid' in kwargs:
            ham, grid = kwargs['hamiltonian'], kwargs['grid']

            print('Finding eigenvalues...')
            eig = np.linalg.eig(ham.mat)

            self.values = np.real_if_close(eig[0], tol=1E7)
            self.grid = grid
            progressbar = ProgressInformer('Evaluating wave functions', length=40)
            i = 0
            self.states = []
            for line in eig[1].transpose():
                self.states.append(WaveFunction(self.grid, line))
                i += 1
                progressbar.report_progress(i / len(eig[1]))
            progressbar.finish()
            self.alias = '{}_{}'.format(
                type(ham).__name__,
                '_'.join('({},{},{})'.format(*b, s) for b, s in zip(grid.bounds, grid.sizes))
            )
            print('Schrodinger equation initialization done\n')
        elif 'filename' in kwargs:
            print('Loading solution from file...')
            self.load(kwargs['filename'])
            print('Schrodinger equation initialization done\n')
        else:
            raise ValueError('Cannot instantiate Solution with arguments given')

    def dump(self, filename: str):
        """
        Dump solution to a CSV file.

        :param filename: File to dump solution to
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        writer = csv.writer(open(filename, 'w'))
        writer.writerow([b[0] for b in self.grid.bounds])
        writer.writerow([b[1] for b in self.grid.bounds])
        writer.writerow(self.grid.sizes)
        writer.writerow(self.values)
        progressbar = ProgressInformer('Dumping wave functions', length=40)
        for i in range(len(self.states)):
            writer.writerow(self.states[i].values)
            progressbar.report_progress((i + 1) / len(self.states))
        progressbar.finish()

    def load(self, filename: str):
        """
        Loads solution from provided CSV file.

        :param filename: File to load solution from
        """
        if not filename.endswith('.csv'):
            raise ValueError('Solution file must have CSV extension')
        self.alias = filename[:-4]
        reader = csv.reader(open(filename))
        l_bounds = [float(a) for a in reader.__next__()]
        u_bounds = [float(a) for a in reader.__next__()]
        sizes = [int(a) for a in reader.__next__()]
        self.grid = Grid(list(zip(l_bounds, u_bounds)), sizes)
        self.values = np.array([complex(a) for a in reader.__next__()])
        self.states = []
        progressbar = ProgressInformer('Loading wave functions', length=40)
        progressbar.report_progress(0)
        for i in range(len(self.values)):
            self.states.append(WaveFunction(self.grid, reader.__next__()))
            progressbar.report_progress((i + 1) / len(self.values))
        progressbar.finish()

    def __getitem__(self, args):
        """
        Obtains states with energy nearest to given.
        If tolerance is specified, energies that differ less are considered equal.

        Takes 1 or 2 arguments.
        Given one argument, takes it as the energy and tolerance as 0.
        Given the second argument, takes it as the tolerance value.

        :return: Tuple with states
        """
        if type(args) == tuple:
            energy, tolerance = args
        else:
            energy = args
            tolerance = 0

        nearest_nrg = min(list(self.values), key=lambda current_energy: abs(energy - current_energy))
        return tuple(self.states[i] for i in range(len(self.values)) if abs(self.values[i] - nearest_nrg) <= tolerance)

    def __iter__(self):
        """
        Returns an iterator with all solutions of the initial equation.
        """
        return iter(self.states)

    def __str__(self):
        """
        Generates an unique alias for the solution
        """
        return self.alias
