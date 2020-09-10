import numpy as np
import numpy.linalg as lin
from Physics.Constants import H
from Visual import PlotItem


class SingleDimPhysicalSystem:
    def __init__(self, mesh_points):
        self.mesh_points = sorted(mesh_points)
        self.grid = np.array([[0.] * len(mesh_points)] * len(mesh_points), dtype=np.float)
        self.eigen_system = None

    def populate_matrix(self, mass, potential_callback):
        m = self.mesh_points
        self.eigen_system = None
        self.grid = np.array([[0.] * len(m)] * len(m), dtype=np.float)
        for i in range(1, len(m) - 1):
            self.grid[i, i] = H ** 2 / mass * (1 / (m[i] - m[i - 1]) + 1 / (m[i + 1] - m[i])) / (
                        m[i + 1] - m[i - 1]) + mass * potential_callback(m[i])
            self.grid[i, i - 1] = - H ** 2 / (mass * (m[i] - m[i - 1]) * (m[i + 1] - m[i - 1]))
            self.grid[i, i + 1] = - H ** 2 / (mass * (m[i + 1] - m[i]) * (m[i + 1] - m[i - 1]))

    def calculate_eigen_system(self):
        self.eigen_system = lin.eigh(self.grid)

    def get_x_range(self):
        return self.mesh_points[0], self.mesh_points[-1]

    def get_eigen_values(self):
        if self.eigen_system is None:
            raise ValueError('Eigenvalues are not calculated yet')
        return PlotItem(list(range(len(self.mesh_points))), sorted(list(self.eigen_system[0])))

    def get_eigen_states(self):
        if self.eigen_system is None:
            raise ValueError('Eigenvalues are not calculated yet')
        return [PlotItem(self.mesh_points, state) for state in
                self.eigen_system[1].transpose()]

    def get_eigen_state_energy(self, energy):
        if self.eigen_system is None:
            raise ValueError('Eigenvalues are not calculated yet')
        threshold = abs(energy - self.eigen_system[0][0])
        index = 0
        for i in range(len(self.eigen_system[0])):
            if threshold > abs(energy - self.eigen_system[0][i]):
                threshold = abs(energy - self.eigen_system[0][i])
                index = i
        data = self.eigen_system[1].transpose()[index]
        return self.eigen_system[0][index], PlotItem(self.mesh_points, data)
