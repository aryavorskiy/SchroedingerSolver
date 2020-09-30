#!/usr/bin/python
import csv

from LinearAlgebraModel.Model.Equation import SchrodingerSolution, WaveFunction
from LinearAlgebraModel.Operators import TorqueOperator
from LinearAlgebraModel.Physics import *
from LinearAlgebraModel.Visual import WaveFunctionVisualizer


def square_grid(r, q, dim=1):
    return Grid([(-r, r)] * dim, [q] * dim)


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
    plotter.plot(title=str(wf.operator_value(hamiltonian)), xlabel='X')


alias = '{}_{}'.format(
    type(hamiltonian).__name__,
    '_'.join('({},{},{})'.format(*b, s) for b, s in zip(sol.grid.bounds, sol.grid.sizes))
)

spectre = sol.spectre(TorqueOperator(sol.grid))
writer = csv.writer(open(alias + '_spectre'))
writer.writerow(spectre)
