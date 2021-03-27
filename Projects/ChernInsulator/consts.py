import numpy as np

sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli_mats = [sx, sy, sz]

z = np.zeros((2, 2))
