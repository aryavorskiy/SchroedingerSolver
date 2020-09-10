class DependentVariable:
    def __init__(self, name: str, derive_callback):
        self.name = name
        self.value = None
        self.derive = derive_callback


class CauchyProblem:
    def __init__(self):
        self.t = None
        self.dependentVariables = {}

    def add_dependent_variable(self, dependent_variable: DependentVariable):
        self.dependentVariables.update({dependent_variable.name: dependent_variable})

    def get_state(self):
        return self.t, {name: self[name].value for name in self}

    def set_state(self, initial_t: float, dependent_vars: dict):
        self.t = initial_t
        if type(dependent_vars) == dict and set(dependent_vars.keys()) == set(self.dependentVariables.keys()):
            for dep in self.dependentVariables.values():
                dep.value = dependent_vars[dep.name]

    def get_derivative(self, name: str):
        return self.dependentVariables[name].derive(self.t,
                                                    {dep.name: dep.value for dep in self.dependentVariables.values()})

    def __getitem__(self, name: str):
        return self.dependentVariables[name]

    def __iter__(self):
        for name in self.dependentVariables.keys():
            yield name
