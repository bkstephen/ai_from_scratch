import numpy as np 
from Neuron import Neuron

class LayerDense:

    def __init__(self, n_inputs, n_neurons):
        tempNeurons = []
        for i in range(0, n_neurons):
            tempNeurons.append(Neuron(n_inputs))
        self.neurons = tempNeurons

    def forward(self, inputs):
        biases = []
        weights = []        
        for neuron in self.neurons:
            weights.append(neuron.weights)                    
            biases.append(neuron.bias)            
        self.output = np.dot(inputs, np.array(weights).T) + biases
