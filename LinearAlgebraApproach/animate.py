from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from Physics.Models import Harmonic
from LinearAlgebraApproach.SingleDimPhysicalSystem import SingleDimPhysicalSystem

mass = 200
physics = Harmonic(100)

mesh = [i * i * 0.0000001 * i for i in range(-100,100)]
system = SingleDimPhysicalSystem(mesh)

print('Generating matrix...')
system.populate_matrix(mass, physics.get_potential)

print('Retrieving eigenstates...')
system.calculate_eigen_system()
eigenstates = system.get_eigen_states()

print('Animating wave functions...')
fig = plt.figure()
ax = plt.axes(ylim=(-1, 1))
line, = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(eigenstates[i].xdata, eigenstates[i].ydata)
    return line,


anim = FuncAnimation(fig, lambda i: animate(i), init_func=init, frames=100, interval=50, blit=True)
anim.save('wf_animate.mp4', writer='ffmpeg')
