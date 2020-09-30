import csv

import numpy as np

from LinearAlgebraModel.Model.BaseOperators import LinearOperator
from LinearAlgebraModel.Model.Grid import Grid
from Utils import ProgressInformer


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
        self.values = np.array(values, dtype=np.complex)
        self.values /= sum(x * np.conj(x) for x in self.values)

    def operator_value(self, operator: LinearOperator):
        """
        Calculate the mean operator value for this wave function

        :param operator: Operator object
        :return: Value
        """
        return np.linalg.multi_dot(
            (self.values.transpose(), operator.get_matrix(), self.values))

    def operator_error(self, operator: LinearOperator):
        """
        Calculates the operator value for this wave function.

        :param operator: Operator object
        :return: Mean square error
        """
        val = self.operator_value(operator)
        op = val * np.eye(len(self.grid)) - operator.get_matrix()
        return np.linalg.multi_dot((self.values.transpose(), op, op, self.values))


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
            eig = np.linalg.eig(ham.get_matrix())

            print('Evaluating wave functions...')
            self.values = np.real_if_close(eig[0], tol=1E7)
            self.grid = grid
            self.states = [WaveFunction(self.grid, line) for line in eig[1].transpose()]
            print('Schrodinger equation initialization done\n')
        elif 'filename' in kwargs:
            print('Loading solution from file...')
            self.load(kwargs['filename'])
            print('Schrodinger equation initialization done\n')
        else:
            raise ValueError('Cannot instantiate Solution object with arguments given')

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

    def spectre(self, operator: LinearOperator):
        spectre = []
        progressbar = ProgressInformer('Evaluating spectre', length=40)
        for i in range(len(self.states)):
            wf = self.states[i]
            if wf.operator_error(operator) < 0.001:
                spectre.append(wf.operator_value(operator))
            progressbar.report_progress((i + 1) / len(self.states))
        progressbar.finish()
        return sorted(spectre)

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
        return tuple(self.states[i] for i in range(len(self.values)) if abs(self.values[i] - nearest_nrg) < tolerance)

    def __iter__(self):
        """
        Returns an iterator with all solutions of the initial equation.
        """
        return iter(self.states)
