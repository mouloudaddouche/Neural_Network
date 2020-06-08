from layer import Layer

#Class Flatten Layer

class FlattenLayer(Layer):
 	
 	# return the input after the flattening
    def forward_propagation(self, input_data):
        self.input = input_data
        #reshape : allows to change the dimensions of the table without changing the data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    # return the output error reshaped in size of inputs
    def backward_propagation(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)
