from abc import ABC, abstractmethod

import numpy as np

from LinearAlgebraModel.Model.Grid import Grid, shift_point

H = 1


def add_second_dif_operator_mat(mat, grid, dim):
    """
    Generates a matrix for the second derivative operator in given grid for given axis and adds it to given matrix.
    Implemented in such way to avoid code duplication.

    :param mat: Initial matrix
    :param grid: Grid
    :param dim: Axis index
    :return: Second derivative operator matrix
    """
    d = grid.grid_step(dim)
    for pt in grid:
        if pt[dim] == 0 or pt[dim] == grid.sizes[dim] - 1:
            continue
        pt_idx = grid.index(pt)
        pt_idx_u = grid.index(shift_point(pt, dim, 1))
        pt_idx_d = grid.index(shift_point(pt, dim, -1))
        mat[pt_idx, pt_idx] -= 2 / d ** 2
        mat[pt_idx, pt_idx_d] += 1 / d ** 2
        mat[pt_idx, pt_idx_u] += 1 / d ** 2
    return mat


def get_second_dif_operator_mat(grid, dim):
    """
    Generates a matrix for the second derivative operator in given grid for given axis.

    :param grid:
    :param dim:
    :return:
    """
    mat = np.zeros((len(grid), len(grid)))
    add_second_dif_operator_mat(mat, grid, dim)
    return mat


def get_laplace_operator_mat(grid):
    """
    Generates a matrix for the Laplace operator in given grid

    :param grid: Grid
    :return: Laplace operator matrix
    """
    mat = np.zeros((len(grid), len(grid)))
    for dim in range(grid.dimensions()):
        add_second_dif_operator_mat(mat, grid, dim)
    return mat


class AbstractOperator(ABC):
    """
    Represents an arbitrary operator
    """

    @abstractmethod
    def get_matrix(self, grid: Grid) -> np.array:
        """
        Generates a matrix for given operator in given grid

        :param grid: Grid
        :return: Hamiltonian operator matrix
        """
        pass


class ParticleHamiltonian(AbstractOperator):
    """
    Implementation of a hamiltonian of a single particle
    """
    m = 1

    def get_matrix(self, grid: Grid) -> np.array:
        potential_mat = - H ** 2 / (self.m * 2) * get_laplace_operator_mat(grid)
        for pt in grid:
            i = grid.index(pt)
            potential_mat[i, i] += self.get_potential(grid.point_to_absolute(pt))
        return potential_mat

    @abstractmethod
    def get_potential(self, x: list) -> float:
        pass
