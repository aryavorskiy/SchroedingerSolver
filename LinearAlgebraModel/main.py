#!/usr/bin/python
from LinearAlgebraModel.Model.Equation import SchroedingerSolution, WaveFunction
from LinearAlgebraModel.Model.Grid import Grid
from LinearAlgebraModel.Physics import *
from LinearAlgebraModel.Visual import WaveFunctionVisualizer


def square_grid(r, q, dim=1):
    return Grid([(-r, r)] * dim, [q] * dim)


hamiltonian = Coulomb(1, 10, -1)
grid = square_grid(5, 40, dim=3)
sol = SchroedingerSolution(hamiltonian=hamiltonian, grid=grid)


def plot(wf: WaveFunction, data_format='pr', color_phase=False):
    plotter = WaveFunctionVisualizer(wf)
    plotter.set_value_data(data_format)
    if color_phase:
        plotter.set_color_data('phs')
    plotter.plot(title=str(wf.operator_value(hamiltonian)), xlabel='X')


sol.dump('{}_{}.csv'.format(
    type(hamiltonian).__name__,
    '_'.join('({},{},{})'.format(*b, s) for b, s in zip(grid.bounds, grid.sizes))
))
