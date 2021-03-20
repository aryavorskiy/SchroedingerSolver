from General.Grid import Grid
from Projects.LinearAlgebraModel.Model.BaseOperators import ParticleHamiltonian

K = 1
Epsilon = 1


class Harmonic(ParticleHamiltonian):
    """
    Represents a multi-dimensional quantum harmonic oscillator.
    """

    def __init__(self, grid: Grid, m, w):
        self.m = m
        self.w = w
        super(Harmonic, self).__init__(grid, m)

    def get_potential(self, x: list) -> float:
        return self.m * self.w ** 2 * sum(t ** 2 for t in x) / 2


class Coulomb(ParticleHamiltonian):
    """
    Represents a charged particle in the electric field of another.
    """

    def __init__(self, grid: Grid, m, q1, q2):
        self.m = m
        self.q1 = q1
        self.q2 = q2
        super(Coulomb, self).__init__(grid, m)

    def get_potential(self, x: list) -> float:
        return K * self.q1 * self.q2 / sum(t ** 2 for t in x) ** 0.5


class LennardJones(ParticleHamiltonian):
    """
    Represents a particle in the Lennard-Jones potential.
    """

    def __init__(self, grid: Grid, m, s):
        self.m = m
        self.s = s
        super(LennardJones, self).__init__(grid, m)

    def get_potential(self, x: list) -> float:
        if len(x) > 1:
            raise ValueError('Dimension mismatch')
        return 4 * Epsilon * ((self.s / x[0]) ** 12 - (self.s / x[0]) ** 6)


class MultipleParticleCoulomb1D(ParticleHamiltonian):
    """
    Represents two single-dimensional particles in the Coulomb potential.
    """

    def __init__(self, grid: Grid, m, q_center, q1, q2):
        self.m = m
        self.Q = q_center
        self.q1 = q1
        self.q2 = q2
        super(MultipleParticleCoulomb1D, self).__init__(grid, m)

    def get_potential(self, x: list) -> float:
        if len(x) != 2:
            raise ValueError('Must contain exactly 2 particles')
        return self.Q * self.q1 / abs(x[0]) + self.Q * self.q2 / abs(x[1]) + self.q1 * self.q2 / abs(x[0] - x[1])
