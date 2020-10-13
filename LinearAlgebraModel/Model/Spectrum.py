import csv

from Utils import ProgressInformer


class Spectrum:
    """
    A structure that contains operator spectrum data and implements methods to analyze it.
    """

    def __init__(self, **kwargs):
        """
        Creates a new Spectrum instance.

        :key solution: Solution object to obtain spectrum from
        :key operators: List of Operators to evaluate
        :key operator: The same if only one operator is needed
        :key filename: Filename to load spectrum data from
        """
        if 'solution' in kwargs:
            sol = kwargs['solution']
            if 'operator' in kwargs:
                ops = [kwargs['operator']]
            elif 'operators' in kwargs:
                ops = kwargs['operators']
            else:
                raise KeyError('No operators passed, cannot create empty spectrum object')

            self.spectrum = {type(operator).__name__: {'values': [], 'errors': []} for operator in ops}
            progressbar = ProgressInformer('Evaluating spectrum', length=40)
            counter = 0
            for wf in sol.states:
                for operator in ops:
                    value, error = wf.operator_value_error(operator)
                    self.spectrum[type(operator).__name__]['values'].append(value)
                    self.spectrum[type(operator).__name__]['errors'].append(error)
                    counter += 1
                    progressbar.report_progress(counter / len(sol.states) / len(ops))
            progressbar.finish()
            self.length = len(sol.states)
        elif 'filename' in kwargs:
            reader = csv.reader(open(kwargs['filename']))
            self.length = int(reader.__next__()[0])
            operator_aliases = reader.__next__()
            self.spectrum = {}
            for alias in operator_aliases:
                self.spectrum.update({alias: {
                    'values': [complex(a) for a in reader.__next__()],
                    'errors': [complex(a).real for a in reader.__next__()]
                }})
        elif '__dict' in kwargs:
            self.spectrum = kwargs['__dict']
            self.length = kwargs['__len']

    def dump(self, filename):
        """
        Dumps spectrum data to file.

        :param filename: Filename to write spectrum data to
        """
        operator_aliases = list(self.spectrum)
        with open(filename, 'w') as spectrum_writer:
            writer = csv.writer(spectrum_writer)
            writer.writerow([self.length])
            writer.writerow(operator_aliases)
            for alias in operator_aliases:
                writer.writerows((self.spectrum[alias]['values'], self.spectrum[alias]['errors']))

    def sift(self, rule):
        """
        Creates new Spectrum instance with states that match given rule.

        :param rule: A function that accepts state data in kwargs format,
         with operator aliases as keys and (value, error) tuples as values.
          Must return True if state matches condition, false otherwise.
        :return: Spectrum instance
        """
        new_spectrum = {alias: {'values': [], 'errors': []} for alias in self.spectrum}
        new_length = 0
        operator_aliases = list(self.spectrum)
        for i in range(self.length):
            kw = {alias: (self.spectrum[alias]['values'][i], self.spectrum[alias]['errors'][i]) for alias in
                  operator_aliases}
            if rule(**kw):
                new_length += 1
                for alias in operator_aliases:
                    new_spectrum[alias]['values'].append(self.spectrum[alias]['values'][i])
                    new_spectrum[alias]['errors'].append(self.spectrum[alias]['errors'][i])
        return Spectrum(__dict=new_spectrum, __len=new_length)
