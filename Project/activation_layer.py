import numpy as np
from layer import Layer

#Class Activation Layer

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activ = activation
        self.activ_prime = activation_prime

    # return input after activation
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activ(self.input)
        return self.output
    
    # return input error dE/dX
    def backward_propagation(self, output_error, learning_rate):
        return np.multiply(self.activ_prime(self.input),output_error)
