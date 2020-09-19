from LinearAlgebraModel.Model.Equation import SchroedingerSolution, WaveFunction
from LinearAlgebraModel.Model.Grid import Grid
from LinearAlgebraModel.Physics import Coulomb
from LinearAlgebraModel.Visual import WaveFunctionVisualizer

hamiltonian = Coulomb(1, 1, -1)
grid = Grid([(-0.5, 0.5), (-0.5, 0.5)], [100, 100])
sol = SchroedingerSolution(hamiltonian, grid)


def plot(wf: WaveFunction, color_phase=False):
    plotter = WaveFunctionVisualizer(wf)
    plotter.set_value_data('pr')
    if color_phase:
        plotter.set_color_data('phs')
    plotter.plot(wf.operator_value(hamiltonian))


plot(sol[-200], color_phase=True)
