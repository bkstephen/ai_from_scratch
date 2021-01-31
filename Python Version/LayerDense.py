import numpy as np 
from Neurons import Neurons

class LayerDense:

    def __init__(self, n_inputs, n_neurons):        
        # Initialize weights and biases
        self.neurons = Neurons(n_inputs, n_neurons)
        self.weights = self.neurons.weights
        self.biases = self.neurons.biases      

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
