import numpy as np
class Neurons:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))