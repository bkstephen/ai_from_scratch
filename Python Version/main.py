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

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
# Print accuracy

print(activation2.output[:5])
print("  Loss: ", loss_function.calculate(activation2.output, y))
print('acc:', accuracy)

### TEST STUFF  ###