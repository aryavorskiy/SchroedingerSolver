#!/usr/bin/python
import csv

from LinearAlgebraModel.Model.Equation import SchrodingerSolution, WaveFunction
from LinearAlgebraModel.Operators.Hamiltonian import *
from LinearAlgebraModel.Operators.Measurement import AngularLaplaceOperator
from LinearAlgebraModel.Visual import WaveFunctionVisualizer


def square_grid(r, q, dim=1):
    step = 1 / (dim * 100)
    return Grid([(-r - step * d, r + step * d) for d in range(dim)], [q] * dim)


def load_values_errors(filename):
    reader = csv.reader(open(filename))
    values = [complex(a) for a in reader.__next__()]
    errors = [complex(a).real for a in reader.__next__()]
    return values, errors


if __name__ == '__main__':
    # grid = square_grid(5, 40, dim=3)
    # hamiltonian = Coulomb(grid, 1, 10, -1)
    # sol = SchrodingerSolution(hamiltonian=hamiltonian, grid=grid)
    sol = SchrodingerSolution(filename='Coulomb_(-5,5,20)_(-5,5,20)_(-5,5,20).csv')
    hamiltonian = Coulomb(sol.grid, 1, 10, -1)


    def plot_solution(wf: WaveFunction, data_format='pr', color_phase=False):
        plotter = WaveFunctionVisualizer(wf)
        plotter.set_value_data(data_format)
        if color_phase:
            plotter.set_color_data('phs')
        plotter.plot(title=str(wf.operator_value_error(hamiltonian)[0]), x_label='X')


    spectrum = sol.spectrum(AngularLaplaceOperator(sol.grid))
    spectrum.dump(sol.alias + '_spectrum')
