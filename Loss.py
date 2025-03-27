import numpy as np

class ErrorFunction:
    def fn(self, X): ...
    def dE_do(self, T, X): ...

class SumSquaredError(ErrorFunction):
    def fn(self, T, O):
        return 0.5 * np.sum((T - O) ** 2)

    def dE_do(self, T, O):
        return (-1) * (T - O)

class MeanSquaredError(ErrorFunction):
    def fn(self, T, O):
        return (1 / T.shape[0]) * np.sum((T - O) ** 2)

    def dE_do(self, T, O):
        return (-2 / T.shape[0]) * (T - O)
    
class BinaryCrossEntropy(ErrorFunction):
    def fn(self, T, O):
        O = np.clip(O, 1e-15, 1 - 1e-15) 
        return -np.mean(T * np.log(O) + (1 - T) * np.log(1 - O))

    def dE_do(self, T, O):
        O = np.clip(O, 1e-15, 1 - 1e-15) 
        return (O - T) / (O * (1 - O))

class CategoricalCrossEntropy(ErrorFunction):
    def fn(self, T, O):
        O = np.clip(O, 1e-15, 1)  
        return -np.sum(T * np.log(O)) / T.shape[0]

    def dE_do(self, T, O):
        O = np.clip(O, 1e-15, 1)  
        return -T / O / T.shape[0]
