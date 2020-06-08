import numpy as np

def tanh(x): # tanh(x)
    return np.tanh(x)

def tanh_prime(x): # tanh'(x) = 1 - tanh(x)^2
    return np.subtract(1,np.power(np.tanh(x) , 2)) 

def sigmoid(x): # sigmoid(x) = (1 / 1 + exp(-x))
    return np.divide(1.0,(np.add(1.0,np.exp(-x))))

def sigmoid_prime(x): # sigmoid'(x) = sigmoide(x) * (1 - sigmoid(x))
    return np.multiply(sigmoid(x),(np.subtract(1,sigmoid(x))))

def rectified_linear_unit(x): # rectified_linear_unit(x) = max(0,x)
    return np.maximum(x,0, x)
    
def rectified_linear_unit_prime(x): # rectified_linear_unit'(x) = 1 if (x>=0) else 0
    return np.where(x > 0, 1.0, 0.0)

def softmax(x): # softmax(x) = exp(x) / sum (exp(x))
    return  np.divide(np.exp(x),np.sum(np.exp(x)))

def softmax_prime(z): # softmax'(x) = softmax(x) * (1-softmax(x)) 
    return softmax(z)*(1-softmax(z))
