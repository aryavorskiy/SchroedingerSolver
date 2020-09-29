import colorsys

import numpy as np
from matplotlib import pyplot as pl

from LinearAlgebraModel.Model.Equation import WaveFunction


def color_by_hue(hue):
    return colorsys.hsv_to_rgb(hue, 1, 1)


class WaveFunctionVisualizer:
    def __init__(self, wf: WaveFunction):
        self.wf = wf
        self.val_data = None
        self.col_data = None
        self.coord_mats = [np.array(self.wf.grid.mesh()) for _ in range(self.wf.grid.dimensions())]
        for i in range(self.wf.grid.dimensions()):
            coord_mat = self.coord_mats[i]
            for j in range(coord_mat.size):
                coord_mat[self.wf.grid[j]] = self.wf.grid.point_to_absolute(self.wf.grid[j])[i]

    def __extract_data(self, format):
        """
        Calculate wave function data to plotable format

        :return: multi-dimensional array
        """
        data = np.array(self.wf.grid.mesh())
        if format == 'value' or format == 'val':
            for pt in self.wf.grid:
                data[pt] = self.wf.values[self.wf.grid.index(pt)]
        elif format == 'phase' or format == 'phs':
            for pt in self.wf.grid:
                data[pt] = np.angle(self.wf.values[self.wf.grid.index(pt)]) / (2 * np.pi)
        elif format == 'prob' or format == 'pr':
            for pt in self.wf.grid:
                data[pt] = self.wf.values[self.wf.grid.index(pt)] * np.conj(self.wf.values[self.wf.grid.index(pt)])
        else:
            raise ValueError('Unknown data extraction argument {}'.format(format))
        return data

    def set_value_data(self, data_type: str):
        """
        Select data to be plotted

        :param data_type: Data format
        :return:
        """
        self.val_data = self.__extract_data(data_type)

    def set_color_data(self, data_type: str):
        """
        Select data to be shown as color

        :param data_type: Data format
        :return:
        """
        data = self.__extract_data(data_type)
        self.col_data = np.array(self.wf.grid.mesh([0, 0, 0]))
        for pt in self.wf.grid:
            self.col_data[pt] = color_by_hue(data[pt])

    def plot(self, title='', xlabel='', ylabel=''):
        """
        Plot selected data

        :param title: Plot title
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
            ax.set_xlabel(xlabel)
            if self.col_data is not None:
                ax.scatter(*args, c=self.col_data)
            else:
                ax.plot(*args)
        elif self.wf.grid.dimensions() == 2:
            ax = pl.axes(projection='3d')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if self.col_data is not None:
                ax.plot_surface(*args, facecolors=self.col_data)
            else:
                ax.plot_surface(*args, cmap='viridis')
        else:
            raise ValueError('Cannot plot {}-dimensional wave function'.format(self.wf.grid.dimensions()))
