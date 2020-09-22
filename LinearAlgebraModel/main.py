from LinearAlgebraModel.Model.Equation import SchroedingerSolution, WaveFunction
from LinearAlgebraModel.Model.Grid import Grid
from LinearAlgebraModel.Physics import Coulomb
from LinearAlgebraModel.Visual import WaveFunctionVisualizer

hamiltonian = Coulomb(1, 1, -1)
grid = Grid([(-0.5, 0.5)], [1000])
sol = SchroedingerSolution(hamiltonian=hamiltonian, grid=grid)


def plot(wf: WaveFunction, color_phase=False):
    plotter = WaveFunctionVisualizer(wf)
    plotter.set_value_data('pr')
    if color_phase:
        plotter.set_color_data('phs')
    plotter.plot(wf.operator_value(hamiltonian))


sol.dump(
    f'{type(hamiltonian).__name__}_{"_".join(f"({b[0]},{b[1]},{s})" for b, s in zip(grid.bounds, grid.sizes))}.csv')
