from abc import abstractmethod

import numpy as np

from LinearAlgebraModel.Model.Grid import Grid, shift_point

H = 1


def add_first_dif_operator_mat(mat, grid, dim, multiplier=1):
    """
    Generates a matrix for the first derivative operator in given grid for given axis,
    multiplies it by the multiplier given and adds it to given matrix.

    Implemented in such way to avoid code duplication.

    :param mat: Initial matrix
    :param grid: Grid object
    :param dim: Axis index
    :param multiplier: Multiplier
    """
    d = grid.grid_step(dim)
    for pt in grid:
        if pt[dim] == grid.sizes[dim] - 1:
            continue
        pt_idx = grid.index(pt)
        pt_idx_u = grid.index(shift_point(pt, dim, 1))
        mat[pt_idx, pt_idx] -= 1 / d * multiplier
        mat[pt_idx, pt_idx_u] += 1 / d * multiplier
    return mat


def get_first_dif_operator_mat(grid, dim):
    """
    Generates a matrix for the second derivative operator in given grid for given axis.

    :param grid: Grid object
    :param dim: Axis index
    :return: Second derivative operator matrix
    """
    mat = np.zeros((len(grid),) * 2)
    add_first_dif_operator_mat(mat, grid, dim)
    return mat


def add_second_dif_operator_mat(mat, grid, dim, multiplier=1):
    """
    Generates a matrix for the second derivative operator in given grid for given axis,
    multiplies it by the multiplier given and adds it to given matrix.

    Implemented in such way to avoid code duplication.

    :param mat: Initial matrix
    :param grid: Grid object
    :param dim: Axis index
    :param multiplier: Multiplier
    """
    d = grid.grid_step(dim)
    for pt in grid:
        if pt[dim] == 0 or pt[dim] == grid.sizes[dim] - 1:
            continue
        pt_idx = grid.index(pt)
        pt_idx_u = grid.index(shift_point(pt, dim, 1))
        pt_idx_d = grid.index(shift_point(pt, dim, -1))
        mat[pt_idx, pt_idx] -= 2 / d ** 2 * multiplier
        mat[pt_idx, pt_idx_d] += 1 / d ** 2 * multiplier
        mat[pt_idx, pt_idx_u] += 1 / d ** 2 * multiplier
    return mat


def get_second_dif_operator_mat(grid, dim):
    """
    Generates a matrix for the second derivative operator in given grid for given axis.

    :param grid: Grid object
    :param dim: Axis index
    :return: Second derivative operator matrix
    """
    mat = np.zeros((len(grid),) * 2)
    add_second_dif_operator_mat(mat, grid, dim)
    return mat


def get_laplace_operator_mat(grid):
    """
    Generates a matrix for the Laplace operator in given grid.

    :param grid: Grid object
    :return: Laplace operator matrix
    """
    mat = np.zeros((len(grid),) * 2)
    for dim in range(grid.dimensions()):
        add_second_dif_operator_mat(mat, grid, dim)
    return mat


class LinearOperator:
    """
    Represents an arbitrary operator
    """

    def __init__(self, grid: Grid, mat: np.array):
        self.grid = grid
        self.mat = mat
        pass

    def __assert_compatibility(self, other):
        if type(self) != type(other):
            raise TypeError('Unsupported operand types: {} and {}'.format(type(self), type(other)))
        if self.grid != other.grid:
            raise ValueError('Both operands should have the same grid')

    def __add__(self, other):
        self.__assert_compatibility(other)
        return LinearOperator(self.grid, self.mat + other.mat)

    def __iadd__(self, other):
        self.__assert_compatibility(other)
        self.mat += other.mat

    def __sub__(self, other):
        self.__assert_compatibility(other)
        return LinearOperator(self.grid, self.mat - other.mat)

    def __isub__(self, other):
        self.__assert_compatibility(other)
        self.mat -= other.mat

    def __mul__(self, other):
        if type(other) in {int, float, complex}:
            return LinearOperator(self.grid, np.dot(self.mat, other))
        self.__assert_compatibility(other)
        return LinearOperator(self.grid, np.dot(self.mat, other.mat))

    def __imul__(self, other):
        self.mat = (other * self).mat


class ParticleHamiltonian(LinearOperator):
    """
    Implementation of a hamiltonian of a single particle.
    """

    def __init__(self, grid: Grid, m, *args):
        self.m = m
        operator_mat = - H ** 2 / (self.m * 2) * get_laplace_operator_mat(grid)
        for pt in grid:
            i = grid.index(pt)
            operator_mat[i, i] += self.get_potential(grid.point_to_absolute(pt))
        super(ParticleHamiltonian, self).__init__(grid, operator_mat)

    @abstractmethod
    def get_potential(self, x: list) -> float:
        """

        :param x:
        :return:
        """
        pass
