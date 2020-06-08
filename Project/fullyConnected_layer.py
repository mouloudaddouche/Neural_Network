import numpy as np
from layer import Layer

#Class FullyConnected Layer

class FullyConnectedLayer(Layer):
    #self = the instance of the class
    #input_size = number of neurons entering the layer
    #output_size = number of neurones output the layer
    
    def __init__(self, input_size, output_size):
        self.input_size=input_size
        self.output_size=output_size
        #initialization of weights table and bias table with random number [0,1[
        self.weight = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros((1, output_size))

    # return output (Y) for an input (X) : Y = (W*X) + B 
    # W = weight
    # B = bias
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.add(np.dot(self.input,self.weight),self.bias) # Y =(W*X)+B 
        return self.output

    # updating of the weight and bias values
    # return input error dE/dX
    def backward_propagation(self, output_error, learning_rate) :
        error_weight = np.dot(self.input.reshape(-1,1), output_error)
        self.weight =  self.weight - (error_weight * learning_rate)
        self.bias = self.bias - (output_error * learning_rate)
        return np.dot(output_error, self.weight.T)


