import colorsys

import numpy as np
from matplotlib import pyplot as pl

from LinearAlgebraModel.Model.Equation import WaveFunction


def color_by_hue(hue):
    return colorsys.hsv_to_rgb(hue, 1, 1)


def plot_any(data, title=''):
    """
    Presents any array of data on a scatter plot

    :param data: List with values to plot
    :param title: Plot title
    """
    pl.cla()
    pl.grid()
    pl.xticks([])
    pl.title(title)
    ax = pl.axes()
    ax.scatter(list(range(len(data))), data)


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
        else:
            raise ValueError('Cannot plot {}-dimensional wave function'.format(self.wf.grid.dimensions()))
