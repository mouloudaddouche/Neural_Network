from layer import Layer
from scipy import signal
import numpy as np

class ConvLayer(Layer):
    # input_shape = (i,j,d)
    # kernel_shape = (m,n)
    # layer_depth = output_depth
    
    def __init__(self, input_shape, kernel_shape, layer_depth):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        #initialization of weights table and bias table with random number [0,1[
        self.weights = np.subtract(np.random.rand(kernel_shape[0], kernel_shape[1], self.input_shape[2], layer_depth),0.5)
        self.bias = np.zeros(layer_depth)

    #return output Y for an input X
    def forward_propagation(self, input):
        self.input = input
        self.output = np.zeros((self.input_shape[0]-self.kernel_shape[0]+1, self.input_shape[1]-self.kernel_shape[1]+1, self.layer_depth))
        for i in range(self.layer_depth):
            for j in range(self.input_shape[2]):
                #we use correlated2d because we have 2 dimensional images
                #correlated2d allows to cross-correlate two 2-dimensional arrays
                #the parameter 'valid' mean that the output consists only of those elements that do not rely on the zero-padding.
                self.output[:,:,i] =  self.output[:,:,i] + signal.correlate2d(self.input[:,:,j], self.weights[:,:,j,i], 'valid') + self.bias[i]
        return self.output
        
    # updating of the weight and bias values
    # return input error dE/dX
    def backward_propagation(self, output_error,learning_rate) :
        #initialization of the value of the array : dInput, dWeights, dBias with 0
        dInput = np.zeros(self.input_shape)
        dWeights = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_shape[2], self.layer_depth))
        dBias = np.zeros(self.layer_depth)

        for i in range(self.layer_depth):
            for j in range(self.input_shape[2]):
                #we use correlated2d / convolve2d because we have 2 dimensional images
                #correlated2d allows to cross-correlate two 2-dimensional arrays
                #convolve2d allows to convolve two 2-dimensional arrays.
                #the parameter 'valid' mean that the output consists only of those elements that do not rely on the zero-padding.
                #we calculate dWeight dE/dW, dBias dE/dB that we use in updating tue values of weight table and bias table
                dInput[:,:,j] = np.add(dInput[:,:,j],signal.convolve2d(output_error[:,:,i], self.weights[:,:,j,i]))
                dWeights[:,:,j,i] = signal.correlate2d(self.input[:,:,j], output_error[:,:,i], 'valid')
            dBias[i] = self.layer_depth * np.sum(output_error[:,:,i])

        #updating the value of weight table and bias table
        self.weights = np.subtract(self.weights , np.multiply(dWeights, learning_rate))
        self.bias = np.subtract(self.bias , np.multiply(dBias,learning_rate))
        #return dInput dE/dX
        return dInput
