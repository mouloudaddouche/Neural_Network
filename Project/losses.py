import numpy as np

#mean squared error loss function
def mean_squared_error(y_true, y_pred) : # MSE (Y,Y') = 1/m *( sum (Y(i)-(Y(i') au carre )
    return np.mean(np.power(np.subtract(y_true,y_pred) , 2))

# the derivate of mean squared error loss function
def mean_squared_error_prime(y_true, y_pred) : # MSE(Y,Y') = ( (Y'-Y)/(size(Y)) ) * 2 
	 return np.multiply(np.divide((np.subtract(y_pred,y_true)),y_true.size),2)
