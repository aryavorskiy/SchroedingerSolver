import colorsys

import numpy as np
from matplotlib import pyplot as pl

from LinearAlgebraModel.Model.Equation import WaveFunction
from LinearAlgebraModel.Model.Spectrum import Spectrum


def color_by_hue(hue):
    return colorsys.hsv_to_rgb(hue, 1, 1)


def plot_polar(wf: WaveFunction, center=None):
    if center is None:
        center = [sum(b) / len(b) for b in wf.grid.bounds]
    elif len(center) != len(wf.grid):
        raise ValueError('Center coordinates must have same dimension number as grid')
    k = []
    v = []
    for pt in wf.grid:
        k.append(sum((pair[0] - pair[1]) ** 2 for pair in zip(wf.grid.point_to_absolute(pt), center)) ** 0.5)
        v.append(abs(wf.values[wf.grid.index(pt)]) ** 2)
    plot_any(v, k)


def plot_spectrum(spectrum: Spectrum, title=''):
    pl.cla()
    pl.grid()
    pl.xticks([])
    pl.title(title)
    ax = pl.axes()
    for alias in spectrum.operators():
        values = [float(e[alias][0].real) for e in spectrum]
        ax.scatter(list(range(len(values))), values, label=alias)
    pl.legend()
    pl.show()


def plot_any(data, keys=None, title=''):
    """
    Presents any array of data on a scatter plot

    :param data: Iterable with values to plot
    :param keys: X values list. Generated from number range automatically if not given
    :param title: Plot title
    """
    pl.cla()
    pl.grid()
    pl.title(title)
    ax = pl.axes()
    if keys is None:
        keys = list(range(len(data)))
        pl.xticks([])
    ax.scatter(keys, data)
    pl.show()


class WaveFunctionVisualizer:
    """
    A structure that contains methods to process and visualize wave function data
    """

    def __init__(self, wf: WaveFunction):
        """
        Creates a new WaveFunctionVisualizer instance.

        :param wf: Wave function to plot
        """
        self.wf = wf
        self.val_data = None
        self.col_data = None
        self.coord_mats = [np.array(self.wf.grid.mesh()) for _ in range(self.wf.grid.dimensions())]
        for i in range(self.wf.grid.dimensions()):
            coord_mat = self.coord_mats[i]
            for j in range(coord_mat.size):
                coord_mat[self.wf.grid[j]] = self.wf.grid.point_to_absolute(self.wf.grid[j])[i]

    def __extract_data(self, data_type):
        """
        Calculate wave function data to plot data format.

        :return: multi-dimensional array
        """
        data = np.array(self.wf.grid.mesh())
        if data_type == 'value' or data_type == 'val':
            for pt in self.wf.grid:
                data[pt] = self.wf.values[self.wf.grid.index(pt)]
        elif data_type == 'phase' or data_type == 'phs':
            for pt in self.wf.grid:
                data[pt] = np.angle(self.wf.values[self.wf.grid.index(pt)]) / (2 * np.pi)
        elif data_type == 'prob' or data_type == 'pr':
            for pt in self.wf.grid:
                data[pt] = self.wf.values[self.wf.grid.index(pt)] * np.conj(self.wf.values[self.wf.grid.index(pt)])
        else:
            raise ValueError('Unknown data extraction argument {}'.format(data_type))
        return data

    def set_value_data(self, data_type: str):
        """
        Select data to be plotted.

        :param data_type: Data format
        :return:
        """
        self.val_data = self.__extract_data(data_type)

    def set_color_data(self, data_type: str):
        """
        Select data to be shown as color.

        :param data_type: Data format
        :return:
        """
        data = self.__extract_data(data_type)
        self.col_data = np.array(self.wf.grid.mesh([0, 0, 0]))
        for pt in self.wf.grid:
            self.col_data[pt] = color_by_hue(data[pt])

    def plot(self, title='', x_label='', y_label=''):
        """
        Plot selected data

        :param title: Plot title
        :param x_label: Text label on the X axis
        :param y_label: Text label on the Y axis
        :return:
        """
        if self.val_data is None:
            raise AttributeError('Value data is not initialized')
        pl.cla()
        pl.grid()
        pl.title(title)
        args = *self.coord_mats, self.val_data
        if self.wf.grid.dimensions() == 1:
            ax = pl.axes()
            ax.set_xlabel(x_label)
            if self.col_data is not None:
                ax.scatter(*args, c=self.col_data)
            else:
                ax.plot(*args)
        elif self.wf.grid.dimensions() == 2:
            ax = pl.axes(projection='3d')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if self.col_data is not None:
                ax.plot_surface(*args, facecolors=self.col_data)
            else:
                ax.plot_surface(*args, cmap='viridis')
            pl.show()
        else:
            raise ValueError('Cannot plot {}-dimensional wave function'.format(self.wf.grid.dimensions()))
