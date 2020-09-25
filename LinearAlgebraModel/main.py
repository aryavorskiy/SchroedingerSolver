#!/usr/bin/python
from LinearAlgebraModel.Model.Equation import SchroedingerSolution, WaveFunction
from LinearAlgebraModel.Model.Grid import Grid
from LinearAlgebraModel.Physics import MultipleParticleCoulomb1D
from LinearAlgebraModel.Visual import WaveFunctionVisualizer


def square_grid(r, q):
    return Grid([(-r * 1.01, r * 1.01), (-r, r)], [q, q])


hamiltonian = MultipleParticleCoulomb1D(10, 1, -1, 1)
grid = square_grid(0.5, 50)
sol = SchroedingerSolution(hamiltonian=hamiltonian, grid=grid)


def plot(wf: WaveFunction, color_phase=False):
    plotter = WaveFunctionVisualizer(wf)
    plotter.set_value_data('pr')
    if color_phase:
        plotter.set_color_data('phs')
    plotter.plot(wf.operator_value(hamiltonian))


sol.dump('{}_{}.csv'.format(
    type(hamiltonian).__name__,
    '_'.join('({},{},{})'.format(*b, s) for b, s in zip(grid.bounds, grid.sizes))
))
