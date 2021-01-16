import numpy as np
class Neuron:
    def __init__(self, n_inputs):
        self.weights = 0.10 * np.random.randn(n_inputs)
        self.bias = 0