import numpy as np


class Regularization:
    def R(self, W): ...
    def before_back_propagation(self, W): ...
    def dR_dw(self, W, w): ...


class LNormRegularization(Regularization):
    p: int
    dR_dW: float

    def __init__(self, p) -> None:
        super().__init__()
        self.p = p

    def before_back_propagation(self, W):
        self.dR_dW = (1 / self.p) * (np.sum(W**self.p) ** ((1 / self.p) - 1))

    def R(self, W):
        return np.sum(W**self.p) ** (1 / self.p)

    def dR_dw(self, w):
        return self.dR_dW * self.p * (w ** (self.p - 1)) * np.sign(w)
