import numpy as np 
from LayerDense import LayerDense
from ActivationReLU import ActivationReLU
from ActivationSoftmax import ActivationSoftmax
from LossCategoricalCrossentropy import LossCategoricalCrossentropy

### TEST STUFF 1 ###
# from data_gen import data_gen

# X, y = data_gen().spiral_data(100, 3)

# layer1 = LayerDense(2,3)
# activation1 = ActivationReLU()
# layer2 = LayerDense(3,3)
# activation2 = ActivationSoftmax()
# layer1.forward(X)
# activation1.forward(layer1.output)
# layer2.forward(activation1.output)
# activation2.forward(layer2.output)
# loss_function = LossCategoricalCrossentropy()

# # Calculate accuracy from output of activation2 and targets
# # calculate values along first axis
# predictions = np.argmax(activation2.output, axis=1)
# if len(y.shape) == 2:
#     y = np.argmax(y, axis=1)
# accuracy = np.mean(predictions == y)
# # Print accuracy

# print(activation2.output[:5])
# print("  Loss: ", loss_function.calculate(activation2.output, y))
# print('acc:', accuracy)

### TEST STUFF 2  ###

# Create dataset
from nnfs.datasets import vertical_data

X, y = vertical_data(samples=100, classes=3)

# Create model
dense1 = LayerDense(2, 3) # first dense layer, 2 inputs
activation1 = ActivationReLU()
dense2 = LayerDense(3, 3) # second dense layer, 3 inputs, 3 outputs
activation2 = ActivationSoftmax()

# Create loss function
loss_function = LossCategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):

    # Update weights with some small random values
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
                'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    # Revert weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()