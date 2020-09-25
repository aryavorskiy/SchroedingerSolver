from LinearAlgebraModel.Model.Operators import ParticleHamiltonian

K = 1
Epsilon = 1


class Harmonic(ParticleHamiltonian):
    """
    Represents a multi-dimensional quantum harmonic oscillator
    """

    def __init__(self, m, w):
        self.m = m
        self.w = w

    def get_potential(self, x: list) -> float:
        return self.m * self.w ** 2 * sum(t ** 2 for t in x) / 2


class Coulomb(ParticleHamiltonian):
    """
    Represents a charged particle in the electric field of another
    """

    def __init__(self, m, q1, q2):
        self.m = m
        self.q1 = q1
        self.q2 = q2

    def get_potential(self, x: list) -> float:
        return K * self.q1 * self.q2 / sum(t ** 2 for t in x) ** 2


class LennardJones(ParticleHamiltonian):
    """
    Represents a particle in the Lennard-Jones potential
    """

    def __init__(self, m, s):
        self.m = m
        self.s = s

    def get_potential(self, x: list) -> float:
        if len(x) > 1:
            raise ValueError('Dimension mismatch')
        return 4 * Epsilon * ((self.s / x[0]) ** 12 - (self.s / x[0]) ** 6)


class MultipleParticleCoulomb1D(ParticleHamiltonian):
    """
    Represents two single-dimensional particles in the Coulomb potential
    """

    def __init__(self, Q, q1, q2, m):
        self.m = m
        self.Q = Q
        self.q1 = q1
        self.q2 = q2

    def get_potential(self, x: list) -> float:
        if len(x) != 2:
            raise ValueError('Should contain exactly 2 particles')
        return self.Q * self.q1 / x[0] + self.Q * self.q2 / x[1] + self.q1 * self.q2 / abs(x[0] - x[1])
