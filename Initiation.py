import numpy as np

class InitationFunction:
    def init(self, shape): ...

class ZeroInitation(InitationFunction):
    def init(self, shape):
        return np.zeros(shape)

class UniformInitation(InitationFunction):
    def init(self, shape, lower_bound=-0.1, upper_bound=0.1, seed=73):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(lower_bound, upper_bound, shape)

class NormalInitation(InitationFunction):
    def init(self, shape, mean=0.0, std=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, std, shape)
