import os

import numpy as np
from matplotlib import pyplot as pl

from General.Utils import ProgressInformer


class Visualizer:
    def __init__(self, coords, datas):
        self.coords = coords
        self.datas = datas

    def plot(self, title='', x_label='', y_label='', filename=None, **kwargs):
        """
        Plot loaded data

        :param title: Plot title
        :param x_label: Text label on the X axis
        :param y_label: Text label on the Y axis
        :param filename: Filename to save image to
        :return:
        """
        if len(self.datas) == 0:
            raise AttributeError('Value data is empty')
        fig = pl.figure()
        if 'quiver' in kwargs and kwargs['quiver']:
            if len(self.coords) != 2:
                raise ValueError('Cannot quiver {}-dimensional data'.format(len(self.coords)))
            if len(self.datas) != 2:
                raise ValueError('Invalid quiver data')
            ax = fig.gca()
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.quiver(*self.coords, *[a[0] for a in self.datas])
        else:
            if len(self.coords) == 1:
                ax = fig.gca()
                ax.set_title(title)
                ax.set_xlabel(x_label)
                for data, label in self.datas:
                    ax.plot(*self.coords, data, label=label)
            elif len(self.coords) == 2:
                ax = fig.gca(projection='3d')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(title)
                for data, label in self.datas:
                    ax.plot_surface(*self.coords, data, cmap='viridis')
            else:
                raise ValueError('Cannot plot {}-dimensional data'.format(len(self.coords)))
        ax.legend()
        if not filename:
            pl.show()
        else:
            pl.savefig(filename)
            pl.gca()
            pl.close(fig)


class FunctionVisualizer(Visualizer):
    """
    A structure that contains methods to process and visualize data
    """

    def __init__(self, grid):
        self.grid = grid
        coords = [np.array(self.grid.mesh()) for _ in range(self.grid.dimensions())]
        for point in self.grid:
            for i in range(self.grid.dimensions()):
                coords[i][point] = self.grid.point_to_absolute(point)[i]
        super().__init__(coords, [])

    def add_fn(self, fn, label=""):
        data = np.array(self.grid.mesh())
        p = ProgressInformer(caption=f'Populating graph for function {label}', max=len(self.grid))
        for point in self.grid:
            data[point] = fn(self.grid.point_to_absolute(point))
            p.report_increment()
        p.finish()
        self.datas.append((data, label))

    def cleanup(self):
        self.datas = []


class ParametricVisualizer(FunctionVisualizer):
    def __init__(self, grid, fns):
        super(ParametricVisualizer, self).__init__(grid)
        self.fns = fns
        self.params = []

    def add_parameter(self, value):
        self.params.append(value)
        for fn in self.fns:
            self.add_fn(lambda data: fn(data, value), f'{fn}: M = {value}')

    def cleanup(self):
        super(ParametricVisualizer, self).cleanup()
        self.params.clear()

    def plot(self, title=None, x_label='', y_label='', filename=None, **kwargs):
        if title is None:
            if len(self.params) == 1:
                title = f'M = {round(float(self.params[0]), 2):0<{5}}'
            else:
                title = f'M in {self.params}'
        super().plot(title, x_label, y_label, filename, **kwargs)

    def plot_value(self, value, filename=None):
        self.cleanup()
        self.add_parameter(value)
        self.plot(filename=filename)

    def animate(self, parameter_values, time, filename, dir_name):
        p = ProgressInformer(caption='Rendering frames...', max=len(parameter_values))
        frame_i = 0
        for M in parameter_values:
            frame_i += 1
            self.plot_value(M, filename=f'{dir_name}/frame_{frame_i:0>{len(parameter_values)}}.png')
            p.report_increment()
        p.finish()

        print('Animating GIF...')
        os.system(f'convert -delay {int(100 * time / len(parameter_values))} {dir_name}/frame_*.png {filename}')
        print('Done')
