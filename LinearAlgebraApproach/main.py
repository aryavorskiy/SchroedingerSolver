#!/usr/bin/python3
from LinearAlgebraApproach.SingleDimPhysicalSystem import SingleDimPhysicalSystem
from Physics.Models import Coulomb
from LinearAlgebraApproach.Mesh import plain_mesh

import matplotlib.pyplot as pl

mass = 10
physics = Coulomb(100, -10)
energy = -3000
r = 5

bounds = (0.01, 5)
step_values = [0.0005]

nrgl_x = []
nrgl_y = []

for step in step_values:
    print(f'Evaluating system for mesh step {step}')

    system = SingleDimPhysicalSystem(plain_mesh(bounds, step=step))

    print('Generating matrix...')
    system.populate_matrix(mass, physics.get_potential)

    print('Retrieving eigenvalues...')
    system.calculate_eigen_system()

    print('Done')

    ground_energy, ground_state = system.get_eigen_state_energy(energy)

    numerically_obtained = list(system.get_eigen_values().ydata)
    idx = numerically_obtained.index(ground_energy)

    nrgl_x.extend([step] * (2 * r + 1))
    nrgl_y.extend(numerically_obtained[idx - r:idx + r + 1])

    print(f'Nearest energy to {energy} is {ground_energy}. Neighbouring energy levels listed below:')
    for i in range(idx - r, idx + r + 1):
        print('>' if i == idx else ' ', numerically_obtained[i])

    ground_state.plot(filename=f'step_{step}.png', title=f'Mesh step: {step}\nState energy: {ground_energy}')
    print()

ax = pl.axes(xlabel='d', ylabel='E')
ax.plot(nrgl_x, nrgl_y, 'bo', ms=2)