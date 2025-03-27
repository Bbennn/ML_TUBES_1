import numpy as np

class InitializationFunction:
    def init(self, shape): ...

class ZeroInitialization(InitializationFunction):
    def init(self, shape):
        return np.zeros(shape)

class UniformInitialization(InitializationFunction):
    def __init__(self, lower_bound=-0.5, upper_bound=0.5, seed=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.random = np.random.RandomState(seed)
    
    def init(self, shape):
        return self.random.uniform(self.lower_bound, self.upper_bound, shape)

class NormalInitialization(InitializationFunction):
    def __init__(self, mean=0.0, std=0.05, seed=None):
        self.mean = mean
        self.std = std
        self.random = np.random.RandomState(seed)
    
    def init(self, shape):
        return self.random.normal(self.mean, self.std, shape)

class XavierInitialization(InitializationFunction):
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed)
    
    def init(self, shape):
        fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return self.random.normal(0.0, std, shape)

class HeInitialization(InitializationFunction):
    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed)
    
    def init(self, shape):
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return self.random.normal(0.0, std, shape)
    
INITIALIZER = {
    "zero": ZeroInitialization,
    "uniform": UniformInitialization,
    "normal": NormalInitialization,
    "xavier": XavierInitialization,
    "he": HeInitialization
}

def get_initializer(initializer_name: str, **kwargs):
    name = initializer_name.lower()
    if name not in INITIALIZER:
        raise ValueError(f"Initializer `{name}` not available")
    return INITIALIZER[name](**kwargs)
