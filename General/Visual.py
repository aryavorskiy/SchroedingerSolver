import os

import numpy as np
from matplotlib import pyplot as pl

from General.Utils import ProgressInformer


class Visualizer:
    """
    A structure that contains methods to process and visualize data
    """

    def __init__(self, grid):
        self.grid = grid
        self.coords = [np.array(self.grid.mesh()) for _ in range(self.grid.dimensions())]
        for point in self.grid:
            for i in range(self.grid.dimensions()):
                self.coords[i][point] = self.grid.point_to_absolute(point)[i]
        self.datas = []

    def add_fn(self, fn, label=""):
        data = np.array(self.grid.mesh())
        for point in self.grid:
            data[point] = fn(self.grid.point_to_absolute(point))
        self.datas.append((data, label))

    def cleanup(self):
        self.datas = []

    def plot(self, title='', x_label='', y_label='', filename=None):
        """
        Plot selected data

        :param title: Plot title
        :param x_label: Text label on the X axis
        :param y_label: Text label on the Y axis
        :param filename: Filename to save image to
        :return:
        """
        if len(self.datas) == 0:
            raise AttributeError('Value data is empty')
        fig = pl.figure()
        if self.grid.dimensions() == 1:
            ax = fig.gca()
            ax.set_title(title)
            ax.set_xlabel(x_label)
            for data, label in self.datas:
                ax.plot(*self.coords, data, label=label)
        elif self.grid.dimensions() == 2:
            ax = fig.gca(projection='3d')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            for data, label in self.datas:
                ax.plot_surface(*self.coords, data, cmap='viridis')
        else:
            raise ValueError('Cannot plot {}-dimensional function'.format(self.grid.dimensions()))
        ax.legend()
        if not filename:
            pl.show()
        else:
            pl.savefig(filename)
            pl.gca()
            pl.close(fig)


class ParametricVisualizer(Visualizer):
    def __init__(self, grid, fns):
        super().__init__(grid)
        self.fns = fns
        self.params = []

    def add_parameter(self, value):
        self.params.append(value)
        for fn in self.fns:
            self.add_fn(lambda data: fn(data, value), f'{fn}: M = {value}')

    def cleanup(self):
        super(ParametricVisualizer, self).cleanup()
        self.params.clear()

    def plot(self, title=None, x_label='', y_label='', filename=None):
        if title is None:
            if len(self.params) == 1:
                title = f'M = {round(float(self.params[0]), 2):0<{5}}'
            else:
                title = f'M in {self.params}'
        super().plot(title, x_label, y_label, filename)

    def plot_value(self, value, filename=None):
        self.cleanup()
        self.add_parameter(value)
        self.plot(filename=filename)

    def animate(self, parameter_values, time, filename, dir_name):
        p = ProgressInformer('Rendering frames...')
        frame_i = 0
        for M in parameter_values:
            frame_i += 1
            self.plot_value(M, filename=f'{dir_name}/frame_{frame_i:0>{len(parameter_values)}}.png')
            p.report_progress(frame_i / len(parameter_values))
        p.finish()

        print('Animating GIF...')
        os.system(f'convert -delay {int(100 * time / len(parameter_values))} {dir_name}/frame_*.png {filename}')
        print('Done')
