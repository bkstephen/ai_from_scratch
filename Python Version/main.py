import numpy as np 
from LayerDense import LayerDense
from ActivationReLU import ActivationReLU


### TEST STUFF  ###
from data_gen import data_gen

X, y = data_gen().spiral_data(100, 3)   

layer1 = LayerDense(2,5)
activation1 = ActivationReLU()

layer1.forward(X)

#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

### TEST STUFF  ###