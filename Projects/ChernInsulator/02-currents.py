import numpy as np

from General.Grid import Grid
from General.Utils import ProgressInformer
from General.Visual import FunctionVisualizer
from Projects.ChernInsulator.consts import pauli_mats, sx, sy, sz, z


def rect(w, h):
    return Grid([[0, w - 1], [0, h - 1]], [w, h])


def generate_h_mat(grid: Grid, m):
    blocks = [[z for _ in range(len(grid))] for _ in range(len(grid))]
    mat = np.zeros((len(grid) * 2,) * 2)
    p = ProgressInformer(caption='Generating matrix...', max=len(grid))
    for pt in grid:
        i = grid.index(pt)
        blocks[i][i] = m * sz
        if pt[0] != grid.sizes[0] - 1:
            i_dx = grid.index(grid.shift_point(pt, 0, 1))
            blocks[i][i_dx] = 0.5 * (sz - 1j * sx)
            blocks[i_dx][i] = 0.5 * (sz + 1j * sx)
        if pt[1] != grid.sizes[1] - 1:
            i_dy = grid.index(grid.shift_point(pt, 1, 1))
            blocks[i][i_dy] = 0.5 * (sz - 1j * sy)
            blocks[i_dy][i] = 0.5 * (sz + 1j * sy)
        p.report_increment()
    p.finish()
    print('Copying...')
    return np.block(blocks)


def current(grid, eig, point, axis):
    point = [int(round(a)) for a in point]
    i = grid.index(point)
    if point[axis] == grid.sizes[axis] - 1:
        return 0
    j = grid.index(grid.shift_point(point, axis, 1))
    s = pauli_mats[axis]
    m = np.block([[z, 0.5 * (sz - 1j * s)], [-0.5 * (sz + 1j * s), z]])
    e = np.concatenate((eig.transpose()[2 * i: 2 * i + 2], eig.transpose()[2 * j: 2 * j + 2]), 0)
    # s = 0
    # for state in e:
    #     s += np.linalg.multi_dot((state.transpose().conjugate(), m, state))
    # return s.imag / len(eig)
    return -np.linalg.multi_dot((e.transpose().conjugate(), m, e)).trace().imag / len(eig)


def current_filtered(grid, eig, point, is_x: bool):
    th = 2
    for i in range(grid.dimensions()):
        if th <= point[i] <= grid.sizes[i] - 2 - th or point[i] == grid.sizes[i] - 1:
            return 0
    return current(grid, eig, point, is_x)


grid = rect(19, 19)
M = 5

H = generate_h_mat(grid, M)

e = np.linalg.eig(H)[1].transpose()[10:20]
v = FunctionVisualizer(grid)

v.add_fn(lambda data: current(grid, e, data, 0), 'j_x')
v.add_fn(lambda data: current(grid, e, data, 1), 'j_y')

# v.plot()
v.plot(quiver=True, title=f'M = {M}')
