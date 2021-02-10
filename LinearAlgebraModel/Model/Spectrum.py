import csv

from LinearAlgebraModel.Model.Equation import naive_operator_value_error
from Utils import ProgressInformer


class SpectrumEntry:
    def __init__(self, tol, **kw):
        self.rel_tolerance = tol
        self.operator_values = {}
        for operator_alias in kw:
            value, error = kw[operator_alias]['value'], kw[operator_alias]['error']
            if type(error) == complex:
                error = float(error.real)
            self.operator_values[operator_alias] = value, error
            setattr(self, operator_alias, (value, error))

    def operators(self):
        return list(self.operator_values)

    def __getitem__(self, item):
        return self.operator_values[item]

    def __gt__(self, other):
        for alias in self.operator_values:
            if abs(self.operator_values[alias][0] - other[alias][0]) > self.rel_tolerance * (abs(
                    self.operator_values[alias][1]) + abs(other[alias][1])):
                return self.operator_values[alias][0].real > other[alias][0].real
        return False


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
        if 'naive' in kwargs:
            naive = kwargs['naive']
        else:
            naive = False

        if 'solution' in kwargs:
            sol = kwargs['solution']
            if 'operator' in kwargs:
                ops = [kwargs['operator']]
            elif 'operators' in kwargs:
                ops = kwargs['operators']
            else:
                raise KeyError('No operators passed, cannot create empty spectrum object')

            progressbar = ProgressInformer('Evaluating spectrum', length=40)
            counter = 0
            self.entries = []
            for wf in sol.states:
                kw = {}
                for operator in ops:
                    value, error = naive_operator_value_error(wf, operator) if naive else wf.operator_value_error(
                        operator)
                    if value is None or error is None:
                        break  # Failed to calculate value / error
                    if abs(error) / 100 > abs(value) > 0.1 or (abs(error) > 1 and abs(value) < 0.1):
                        break  # Too large error
                    kw[type(operator).__name__] = {'value': value, 'error': error}
                else:
                    self.entries.append(SpectrumEntry(1, **kw))
                counter += 1
                progressbar.report_progress(counter / len(sol.states))
            progressbar.finish()
        elif 'filename' in kwargs:
            with open(kwargs['filename']) as spectrum_reader:
                reader = csv.reader(spectrum_reader)
                length = int(reader.__next__()[0])
                operator_aliases = reader.__next__()
                data_dict = {}
                for alias in operator_aliases:
                    data_dict.update({alias: {
                        'values': [complex(a) for a in reader.__next__()],
                        'errors': [complex(a).real for a in reader.__next__()]
                    }})
                self.entries = [SpectrumEntry(1, **{alias: {
                    'value': data_dict[alias]['values'][i],
                    'error': data_dict[alias]['errors'][i]}
                    for alias in data_dict}) for i in range(length)]
        elif '__list' in kwargs:
            self.entries = kwargs['__list']

    def operators(self):
        return self.entries[0].operators() if len(self.entries) != 0 else []

    def dump(self, filename):
        """
        Dumps spectrum data to file.

        :param filename: Filename to write spectrum data to
        """
        with open(filename, 'w') as spectrum_writer:
            writer = csv.writer(spectrum_writer)
            writer.writerow([len(self)])
            writer.writerow(self.operators())
            for alias in self.operators():
                values = [e[alias][0] for e in self.entries]
                errors = [e[alias][1] for e in self.entries]
                writer.writerows((values, errors))

    def slice(self, rule):
        """
        Creates new Spectrum instance with states that match given rule.

        :param rule: A function that accepts state data as SpectrumEntry object.
          Must return True if state matches condition, False otherwise.
        :return: Spectrum instance
        """
        new_list = [e for e in self.entries if rule(e)]
        return Spectrum(__list=new_list)

    def sort(self, rel_tolerance=1):
        """
        Sorts spectrum entries according to their eigenvalues.

        :param rel_tolerance: Tolerance coefficient
        """
        for e in self.entries:
            e.rel_tolerance = rel_tolerance
        self.entries.sort()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, operator_alias):
        return [e[operator_alias][0] for e in self.entries], [e[operator_alias][1] for e in self.entries]

    def __iter__(self):
        return iter(self.entries)
