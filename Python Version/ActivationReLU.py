import numpy as np
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)