import matplotlib.pyplot as pl
import numpy as np

from General.Grid import Grid
from General.Utils import ProgressInformer
from General.Visual import FunctionVisualizer
from Projects.ChernInsulator.consts import pauli_mats, sx, sy, sz, z

pl.ioff()


def rect(w, h):
    return Grid([[0, w - 1], [0, h - 1]], [w, h])


def generate_h_mat(grid: Grid, m):
    mat = np.zeros((len(grid) * 2,) * 2, dtype='complex64')
    p = ProgressInformer(caption='Generating matrix...', max=len(grid))
    for pt in grid:
        i = grid.index(pt) * 2
        mat[i:i + 2, i:i + 2] = m * sz
        if pt[0] != grid.sizes[0] - 1:
            i_dx = grid.index(grid.shift_point(pt, 0, 1)) * 2
            mat[i:i + 2, i_dx:i_dx + 2] = 0.5 * (sz - 1j * sx)
            mat[i_dx:i_dx + 2, i:i + 2] = 0.5 * (sz + 1j * sx)
        if pt[1] != grid.sizes[1] - 1:
            i_dy = grid.index(grid.shift_point(pt, 1, 1)) * 2
            mat[i:i + 2, i_dy:i_dy + 2] = 0.5 * (sz - 1j * sy)
            mat[i_dy:i_dy + 2, i:i + 2] = 0.5 * (sz + 1j * sy)
        p.report_increment()
    p.finish()
    return mat


def current(grid, states: np.array, point, axis):
    if len(states.shape) == 1:
        states = np.array([states])
    point = [int(round(a)) for a in point]
    i = grid.index(point)
    if point[axis] == grid.sizes[axis] - 1:
        return 0
    j = grid.index(grid.shift_point(point, axis, 1))
    s = pauli_mats[axis]
    m = np.block([[z, 0.5 * (sz - 1j * s)], [-0.5 * (sz + 1j * s), z]])
    e = np.concatenate((states.transpose()[2 * i: 2 * i + 2], states.transpose()[2 * j: 2 * j + 2]), 0)
    return -np.linalg.multi_dot((e.transpose().conjugate(), m, e)).trace().imag / len(states)


site_grid = rect(30, 30)
M = 5

H = generate_h_mat(site_grid, M)

print(f'Finding eigenvalues [Matrix size {"x".join(str(dim) for dim in H.shape)}] ...', end=' ')
full_eig = np.linalg.eigh(H)
all_states = full_eig[1].transpose()
print('done.')


# states = np.array([s for s in all_states if np.linalg.multi_dot((s.transpose().conjugate(), H, s)) > 0])

def get_energy(state):
    return np.linalg.multi_dot((state.transpose().conjugate(), H, state)).real


def plot_states(data, title_suffix=''):
    title = f'M = {M}'
    if type(data) == int:
        req_states = all_states[data]
        title += f'\nEnergy: {get_energy(req_states):.5f}'
    else:
        try:
            iterator = iter(data)
            req_states = []
            for i in iterator:
                if type(i) == int:
                    req_states.append(all_states[i])
                else:
                    req_states.append(i)
            req_states = np.array(req_states)
        except TypeError:
            raise
    if title_suffix != '':
        title += f'\n{title_suffix}'
    v = FunctionVisualizer(site_grid)
    v.add_fn(lambda pt: current(site_grid, req_states, pt, 0), 'j_x')
    v.add_fn(lambda pt: current(site_grid, req_states, pt, 1), 'j_y')
    v.plot(quiver=True, title=title)


plot_states(1)
