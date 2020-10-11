import csv


class Spectrum:
    """
    A structure that contains operator spectrum data and implements methods to analyze it.
    """

    def __init__(self, **kwargs):
        """
        Creates a new Spectrum instance.

        :key values: List of operator values
        :key errors: List of operator errors
        :key filename: Filename to load spectrum data from
        """
        if 'values' in kwargs:
            self.values = kwargs['values']
            if type(self.values) != list:
                raise TypeError('values argument should be of type list, not {}'.format(type(self.values)))
            self.errors = kwargs['errors'] if 'errors' in kwargs else [0.] * len(self.values)
            if type(self.errors) != list:
                raise TypeError('errors argument should be of type list, not {}'.format(type(self.values)))
        if 'filename' in kwargs:
            reader = csv.reader(open(kwargs['filename']))
            self.values = [complex(a) for a in reader.__next__()]
            self.errors = [complex(a).real for a in reader.__next__()]
        if len(self.values) != len(self.errors):
            raise ValueError('values and errors fields should have the same length')

    def dump(self, filename):
        """
        Dumps spectrum data to file.

        :param filename: Filename to write spectrum data to
        """
        with open(filename) as spectrum_writer:
            writer = csv.writer(spectrum_writer)
            writer.writerows((self.values, self.errors))

    def __getitem__(self, item):
        """
        Selects values that match given rule.

        :param item: Array slice. Start and stop define value bounds, step defines error tolerance
        :return: Spectrum object
        """
        if type(item) == slice:
            matching_criteria = []
            matching_criteria_errs = []
            for i in range(len(self.values)):
                if (item.start is None or self.values[i].real >= item.start) and (
                        item.stop is None or self.values[i].real < item.stop) and (
                        item.step is None or abs(self.errors[i]) < item.step):
                    matching_criteria.append(self.values[i])
                    matching_criteria_errs.append(self.errors[i])
            return Spectrum(values=matching_criteria, errors=matching_criteria_errs)
        else:
            raise TypeError('Argument type must be slice, not {}'.format(type(item)))
