from abc import ABC, abstractmethod
from math import isnan

from General.Utils import ProgressInformer
from Projects.CauchyTaskModel.CauchyProblem import CauchyProblem


class Solver(ABC):
    def __init__(self, problem: CauchyProblem):
        self.problem = problem

    @abstractmethod
    def perform_step(self, _step):
        return {}

    def set_condition(self, initial_t, dependent_vars):
        self.problem.set_state(initial_t, dependent_vars)

    def evolve(self, target, step, verbose=False):
        if step <= 0:
            raise ValueError('Step must be strictly positive')
        initial_t, dependent_vars = self.problem.get_state()
        t = initial_t
        if target < t:
            step *= -1
        t_arr = [t]
        dvars = {name: [dependent_vars[name]] for name in dependent_vars}

        max_progress = int(abs(target - t) / abs(step))
        if verbose:
            informer = ProgressInformer(maximum=max_progress)

        while t * step <= target * step:
            step_result = self.perform_step(step)
            for var in step_result:
                dvars[var].append(step_result[var])
                if isnan(step_result[var]):
                    raise ValueError('Reached NotANumber')
            t_arr.append(t)
            t += step
            if verbose:
                informer.report_increment()
            self.problem.set_state(t, step_result)
        self.problem.set_state(initial_t, dependent_vars)
        if verbose:
            informer.finish()
        return t_arr, dvars

    def evolve_iterative(self, target_t, point_tolerance):
        try:
            tol = float(point_tolerance)

            def point_tolerance_callback(a, b):
                return abs(a - b) < tol
        except TypeError:
            point_tolerance_callback = point_tolerance
        initial_t, dependent_vars = self.problem.get_state()
        step = abs(initial_t - target_t) / 100
        self.set_condition(initial_t, dependent_vars)

        t_arr, dvars_arrs = self.evolve(target_t, step)
        is_desired_accuracy = False
        while not is_desired_accuracy:
            step /= 2
            new_t_arr, new_dvars_arrs = self.evolve(target_t, step)
            is_desired_accuracy = True
            for var in self.problem:
                if not is_desired_accuracy:
                    break
                for i in range(len(dvars_arrs[var]) - 1):
                    if not point_tolerance_callback(dvars_arrs[var][i], new_dvars_arrs[var][2 * i]):
                        t_arr = new_t_arr
                        dvars_arrs = new_dvars_arrs
                        is_desired_accuracy = False
                        break
        return t_arr, dvars_arrs


class SimpleSolver(Solver):
    def __init__(self, problem):
        super().__init__(problem)

    def perform_step(self, step):
        return {name: self.problem.dependentVariables[name].value + step * self.problem.get_derivative(name)
                for name in self.problem}


class RungeKuttaSolver(Solver):
    def __init__(self, problem):
        super().__init__(problem)

    def perform_step(self, step):
        t, dvars = self.problem.get_state()
        k1 = {name: step * self.problem[name].derive(t, dvars) for name in self.problem}
        v1 = {name: dvars[name] + k1[name] / 2 for name in self.problem}
        k2 = {name: step * self.problem[name].derive(t + step / 2, v1) for name in self.problem}
        v2 = {name: dvars[name] + k2[name] / 2 for name in self.problem}
        k3 = {name: step * self.problem[name].derive(t + step / 2, v2) for name in self.problem}
        v3 = {name: dvars[name] + k3[name] for name in self.problem}
        k4 = {name: step * self.problem[name].derive(t + step, v3) for name in self.problem}
        return {name: dvars[name] + (k1[name] + 2 * k2[name] + 2 * k3[name] + k4[name]) / 6 for name in self.problem}
