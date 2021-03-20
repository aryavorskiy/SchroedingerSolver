import os
from math import sin, cos, pi

import matplotlib.pyplot as pl
import numpy as np

from General.Grid import Grid
from General.Visual import ParametricVisualizer

pl.ioff()

# Pauli matrices
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli_mats = [sx, sy, sz]


def get_energy(k, m):
    mat = sz * (m + sum(cos(k_) for k_ in k)) + sum(sin(k[i]) * pauli_mats[i] for i in range(len(k)))
    return sorted(np.linalg.eigvals(mat))


START = -5
END = 5
COUNT = 200
TIME = 5
grid = Grid([[-pi, pi]] * 2, [50] * 2)
fns = [(lambda k, m: get_energy(k, m)[0]), (lambda k, m: get_energy(k, m)[1])]

print('Getting ready...')
anim_i = 1
while os.path.exists(f'anim_{anim_i}'):
    anim_i += 1
os.mkdir(f'anim_{anim_i}')

Ms = [START + (END - START) / COUNT * a for a in range(COUNT)]

vis = ParametricVisualizer(grid, fns)
vis.animate(Ms, TIME, f'anim_{anim_i}/animation.gif', f'anim_{anim_i}')
