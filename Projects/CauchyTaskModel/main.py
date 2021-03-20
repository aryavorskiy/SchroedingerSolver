from Projects.CauchyTaskModel.CauchyProblem import DependentVariable, CauchyProblem
from Projects.CauchyTaskModel.Solver import RungeKuttaSolver

r = 10
h = 1e-34
m = 1e-20
M = m
G = 6.6e-11

p = CauchyProblem()
s = RungeKuttaSolver(p)


def psi_sec_diff(_t, _dvars):
    psi = _dvars['psi']
    return (2 * m / h ** 2) * (G * M * m * (-1 / _t + 1 / r)) * psi


p.add_dependent_variable(DependentVariable('psi', lambda _t, _dvars: _dvars['psi\'']))
p.add_dependent_variable(DependentVariable('psi\'', psi_sec_diff))
p.set_state(1000, {'psi': 1e-100, 'psi\'': -1e-100})

t, dvars = s.evolve(1e-3, 1e-3, True)
dvars.pop('psi\'')
# show_dependents(t, dvars)
# show_dependents(t, dvars, (0, 10))
