import numpy as np

from LinearAlgebraModel.Model.BaseOperators import LinearOperator, get_first_dif_operator_mat
from LinearAlgebraModel.Model.Grid import Grid


class TorqueOperator(LinearOperator):
    def __init__(self, grid: Grid, *args):
        if grid.dimensions() != 3:
            raise ValueError('Grid must be three-dimensional')
        dx_mat = get_first_dif_operator_mat(grid, 0)
        dy_mat = get_first_dif_operator_mat(grid, 1)
        x_mat = np.diagflat(list(pt[0] for pt in grid))
        y_mat = np.diagflat(list(pt[1] for pt in grid))
        super(TorqueOperator, self).__init__(grid, -1j * (np.dot(x_mat, dy_mat) - np.dot(y_mat, dx_mat)))

    def get_matrix(self) -> np.array:
        return self.mat
