import numpy as np

from General.Grid import Grid
from Projects.LinearAlgebraModel.Model.BaseOperators import LinearOperator, ScalarLinearOperator, \
    get_first_dif_operator_mat, H


class TorqueOperator(LinearOperator):
    def __init__(self, grid: Grid, axis_no=2):
        if grid.dimensions() != 3:
            raise ValueError('Grid must be three-dimensional')
        x = (axis_no + 1) % 3
        y = (axis_no + 2) % 3
        dx_mat = get_first_dif_operator_mat(grid, x)
        dy_mat = get_first_dif_operator_mat(grid, y)
        x_mat = np.diagflat(list(grid.point_to_absolute(pt)[x] for pt in grid))
        y_mat = np.diagflat(list(grid.point_to_absolute(pt)[y] for pt in grid))
        super(TorqueOperator, self).__init__(grid, -H * 1j * (np.dot(x_mat, dy_mat) - np.dot(y_mat, dx_mat)))


class TorqueSquaredOperator(LinearOperator):
    def __init__(self, grid: Grid):
        if grid.dimensions() != 3:
            raise ValueError('Grid must be three-dimensional')
        components = [TorqueOperator(grid, axis) for axis in range(3)]
        mat = sum((c * c).mat for c in components)
        super().__init__(grid, mat)


class AngularLaplaceOperator(LinearOperator):
    def __init__(self, grid: Grid):
        mat = - TorqueSquaredOperator(grid).mat * ScalarLinearOperator(grid, lambda x: 1 / sum(a ** 2 for a in x)).mat
        super(AngularLaplaceOperator, self).__init__(grid, mat)
