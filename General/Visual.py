import os

import numpy as np
from matplotlib import pyplot as pl

from General.Utils import ProgressInformer


class Visualizer:
    def __init__(self, coords, datas):
        self.coords = coords
        self.datas = datas
        self.has_legend = False

    def plot(self, **kwargs):
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
            ax.set_title(kwargs.get('title', ''))
            ax.set_xlabel(kwargs.get('x_label', ''))
            ax.quiver(*self.coords, *[a[0] for a in self.datas])
        else:
            if len(self.coords) == 1:
                ax = fig.gca()
                ax.set_title(kwargs.get('title', ''))
                ax.set_xlabel(kwargs.get('x_label', ''))
                for data, label in self.datas:
                    ax.plot(*self.coords, data, label=label)
                if self.has_legend:
                    ax.legend()
            elif len(self.coords) == 2:
                ax = fig.gca(projection='3d')
                ax.set_xlabel(kwargs.get('x_label', ''))
                ax.set_ylabel(kwargs.get('y_label', ''))
                ax.set_title(kwargs.get('title', ''))
                for data, label in self.datas:
                    ax.plot_surface(*self.coords, data, cmap='viridis')
            else:
                raise ValueError('Cannot plot {}-dimensional data'.format(len(self.coords)))
        if 'filename' not in kwargs:
            pl.show()
        else:
            pl.savefig(kwargs['filename'])
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

    def add_fn(self, fn, label="", verbose=True):
        data = np.array(self.grid.mesh())
        p = ProgressInformer(caption=f'Populating graph for function {label}', max=len(self.grid), verbose=verbose)
        for point in self.grid:
            data[point] = fn(self.grid.point_to_absolute(point))
            p.report_increment()
        p.finish()
        if label != "":
            self.has_legend = True
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
            self.add_fn(lambda data: fn(data, value), f'{fn}: M = {value}', verbose=False)

    def cleanup(self):
        super(ParametricVisualizer, self).cleanup()
        self.params.clear()

    def plot(self, **kwargs):
        if 'title' not in kwargs is None:
            if len(self.params) == 1:
                kwargs['title'] = f'M = {self.params[0]:.2f}'
            else:
                kwargs['title'] = f'M in {self.params}'
        super().plot(**kwargs)

    def plot_value(self, value, **kwargs):
        self.cleanup()
        self.add_parameter(value)
        self.plot(**kwargs)

    def animate(self, parameter_values, time, filename, dir_name, draw_args):
        p = ProgressInformer(caption='Rendering frames...', max=len(parameter_values))
        frame_i = 0
        for M in parameter_values:
            frame_i += 1
            self.plot_value(M, filename=f'{dir_name}/frame_{frame_i:0>{len(str(len(parameter_values)))}}.png',
                            **draw_args)
            p.report_increment()
        p.finish()

        print('Animating...')
        os.system(f'ffmpeg -framerate {len(parameter_values) / time:.2f} -i {dir_name}/frame_%03.png {filename}')
        print('Done')
