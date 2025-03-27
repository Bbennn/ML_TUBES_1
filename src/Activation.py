import numpy as np
class ActivationFunction:
    def fn(self, X): ...
    def do_ds(self, nets): ...


class LinearActivationFunction(ActivationFunction):
    def fn(self, N):
        return N

    def do_ds(self, N, O):
        return np.ones(O.shape)


class ReluActivationFunction(ActivationFunction):
    def fn(self, N):
        return np.vectorize(lambda x: 0 if x < 0 else x)(N)

    def do_ds(self, N, O):
        return np.vectorize(lambda x: 0 if x < 0 else 1)(N)


class SigmoidActivationFunction(ActivationFunction):
    def fn(self, N):
        return np.ones(N.shape) / (np.ones(N.shape) + np.exp(-N))

    def do_ds(self, N, O):
        return O * (np.ones(O.shape) - O)


class TanhActivationFunction(ActivationFunction):
    def fn(self, N):
        return np.tanh(N)

    def do_ds(self, N, O):
        return np.reciprocal(np.cosh(N) * np.cosh(N))