import numpy as np 
from LayerDense import LayerDense
from ActivationReLU import ActivationReLU
from ActivationSoftmax import ActivationSoftmax
from LossCategoricalCrossentropy import LossCategoricalCrossentropy

### TEST STUFF  ###
from data_gen import data_gen
X, y = data_gen().spiral_data(100, 3)   

layer1 = LayerDense(2,3)

activation1 = ActivationReLU()

layer2 = LayerDense(3,3)

activation2 = ActivationSoftmax()

layer1.forward(X)

activation1.forward(layer1.output)

layer2.forward(activation1.output)

activation2.forward(layer2.output)

loss_function = LossCategoricalCrossentropy()

print(activation2.output[:5])
print("  Loss: ", loss_function.calculate(activation2.output, y))

### TEST STUFF  ###