#!/usr/bin/python
from LinearAlgebraModel.Model.Equation import SchrodingerSolution, WaveFunction
from LinearAlgebraModel.Model.Spectrum import Spectrum
from LinearAlgebraModel.Operators.Hamiltonian import *
from LinearAlgebraModel.Operators.Measurement import AngularLaplaceOperator, TorqueOperator
from LinearAlgebraModel.Visual import WaveFunctionVisualizer


def square_grid(r, q, dim=1, center: tuple = None):
    if center is None:
        center = (0,) * dim
    step = 1 / (dim * 100)
    return Grid([(-r - step * d, r + step * d) for d in range(dim)], [q] * dim)


if __name__ == '__main__':
    grid = square_grid(5, 4, dim=3)
    hamiltonian = Coulomb(grid, 1, 1, -1)
    sol = SchrodingerSolution(hamiltonian=hamiltonian, grid=grid)


    # sol = SchrodingerSolution(filename='Coulomb_(-5,5,20)_(-5,5,20)_(-5,5,20).csv')
    # hamiltonian = Coulomb(sol.grid, 1, 10, -1)

    def plot_solution(wf: WaveFunction, data_format='pr', color_phase=False):
        plotter = WaveFunctionVisualizer(wf)
        plotter.set_value_data(data_format)
        if color_phase:
            plotter.set_color_data('phs')
        plotter.plot(title=str(wf.operator_value_error(hamiltonian)[0]), x_label='X')


    spectrum = Spectrum(solution=sol, operators=(AngularLaplaceOperator(sol.grid), TorqueOperator(sol.grid)))
    spectrum.dump(sol.alias + '_spectrum')
