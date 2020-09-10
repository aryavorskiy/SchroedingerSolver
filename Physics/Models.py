from abc import ABC, abstractmethod

from Physics.Constants import H, K, Epsilon


class AbstractPhysics(ABC):
    @abstractmethod
    def get_potential(self, x):
        pass

    @abstractmethod
    def get_energy_level(self, i):
        pass


class Harmonic(AbstractPhysics):
    def __init__(self, w):
        self.w = w

    def get_potential(self, x):
        return self.w ** 2 * x ** 2 / 2

    def get_energy_level(self, i):
        return H * self.w * (i + 0.5)


class Coulomb(AbstractPhysics):
    def __init__(self, q1, q2):
        self.q1 = q1
        self.q2 = q2

    def get_potential(self, x):
        return K * self.q1 * self.q2 / x

    def get_energy_level(self, i):
        return 0


class LennardJones(AbstractPhysics):
    def __init__(self, s):
        self.s = s

    def get_potential(self, x):
        return 4 * Epsilon * ((self.s / x) ** 12 - (self.s / x) ** 6)

    def get_energy_level(self, i):
        return 0
