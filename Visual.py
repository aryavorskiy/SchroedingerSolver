from matplotlib import pyplot as pl


class PlotItem:
    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata
        self.lower = xdata[0]
        self.len = xdata[-1] - xdata[0]

    def center(self):
        self.lower = -self.len / 2

    def range(self, lower, upper):
        if upper < lower:
            lower, upper = upper, lower
        self.lower, self.len = lower, upper - lower

    def plot(self, filename=None, title='', range_override=None):
        pl.cla()
        ax = pl.axes()
        ax.plot(self.xdata, self.ydata)
        if range_override is not None:
            pl.xlim(range_override)
        else:
            pl.xlim((self.lower, self.lower + self.len))
        pl.title(title)
        pl.grid()
        if filename is None:
            pl.show()
        else:
            pl.savefig(filename)

    def __getitem__(self, item):
        return self.ydata[item]


def show_dependents(t_array: list, variables: dict, title='', xrange=None):
    fig, ax = pl.subplots()
    for var in variables:
        ax.plot(t_array, variables[var], label=var)
    ax.legend()
    if xrange is not None:
        pl.xlim(xrange)
    pl.title(title)
    pl.grid()
    pl.show()
