#coding:utf-8
import numpy as np
import pandas as pd
from fullyConnected_layer import FullyConnectedLayer
from convolutional_layer import ConvLayer
from flatten_layer import FlattenLayer
from activation_layer import ActivationLayer
from activations import *
from Preprocessing_img import *
from preprocessing_text import get_attributes
from globals import length_m

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

   #this function allows to divide the dataset into two parts : training set and test set
   #The instances are chosen randomly
    def random_split(self, dataset, train_size) :
        if isinstance(dataset,pd.DataFrame) :
          df=dataset
          df.sample(frac=1)
          attributes = get_attributes(df)
          att_class=attributes[df.shape[1]-1]
          del attributes[df.shape[1]-1] 
          X=[]  
          Y=[]
          for i in range(df.shape[0]):
            tmp=[]
            for att in attributes:
              tmp.append(df[att][i])
            X.append(tmp)
          for val in df[att_class]:
            Y.append(val)
          X=np.array(X)
          Y=np.array(Y)
          Y1=np.zeros((Y.shape[0],np.amax(Y)+1),dtype=int) 
          for i in range (Y1.shape[0]): Y1[i:i+1][0,Y[i:i+1]]=1
          X_train = X[:int(X.shape[0]*train_size)]
          Y_train = Y1[:int(Y1.shape[0]*train_size)]
          X_test = X[int(X.shape[0]*train_size):]
          Y_test = Y1[int(Y1.shape[0]*train_size):]
          return ((X_train,Y_train),(X_test,Y_test))
        elif isinstance(dataset,np.ndarray) :
          data=dataset
          data=np.array(data)
          np.arange(data.shape[0]*data.shape[1]).reshape(-1,data.shape[1])
          np.random.shuffle(data)
          X = []
          y = []
          for features,label in data:
             X.append(features)
             y.append(label)
          X = np.array(X).reshape(-1, len(data[0][0]), len(data[0][0]), 1)  #-1==> val deduite par python, #1==> c est en nuances de gris
          X= np.array(X)
          Y= np.array(y)
          Y1=np.zeros((Y.shape[0],np.amax(Y)+1),dtype=int) 
          for i in range (Y1.shape[0]): Y1[i:i+1][0,Y[i:i+1]]=1
          X_train = X[:int(X.shape[0]*train_size)]
          Y_train = Y1[:int(Y1.shape[0]*train_size)]
          X_test = X[int(X.shape[0]*train_size):]
          Y_test = Y1[int(Y1.shape[0]*train_size):]
          return ((X_train,Y_train),(X_test,Y_test))
        else :
          X= np.copy(dataset[0])
          Y= np.copy(dataset[1])
          X_train = X[:int(X.shape[0]*train_size)]
          Y_train = Y[:int(Y.shape[0]*train_size)]
          X_test = X[int(X.shape[0]*train_size):]
          Y_test = Y[int(Y.shape[0]*train_size):]
          return ((X_train,Y_train),(X_test,Y_test))

	 #this function allows to divide the dataset into two parts : training set and test set
   #The instances are chosen regularly
    def regular_split(self, dataset, train_size):
      if isinstance(dataset,pd.DataFrame) :
          df=dataset
          attributes = get_attributes(df)
          att_class=attributes[df.shape[1]-1]
          del attributes[df.shape[1]-1] 
          X=[]  
          Y=[]
          for i in range(df.shape[0]):
            tmp=[]
            for att in attributes:
              tmp.append(df[att][i])
            X.append(tmp)
          for val in df[att_class]:
            Y.append(val)
          X=np.array(X)
          Y=np.array(Y)
          Y1=np.zeros((Y.shape[0],np.amax(Y)+1),dtype=int) 
          for i in range (Y1.shape[0]): Y1[i:i+1][0,Y[i:i+1]]=1
          X_train = X[:int(X.shape[0]*train_size)]
          Y_train = Y1[:int(Y1.shape[0]*train_size)]
          X_test = X[int(X.shape[0]*train_size):]
          Y_test = Y1[int(Y1.shape[0]*train_size):]
          return ((X_train,Y_train),(X_test,Y_test))
      elif isinstance(dataset,np.ndarray) :
            data=dataset
            data=np.array(data)
            np.arange(data.shape[0]*data.shape[1]).reshape(-1,data.shape[1])
            X = []
            y = []
            for features,label in dataset:
               X.append(features)
               y.append(label)
            X = np.array(X).reshape(-1, len(data[0][0]), len(data[0][0]), 1)  #-1==> val deduite par python, #1==> c est en nuances de gris
            X= np.array(X)
            Y= np.array(y)
            Y1=np.zeros((Y.shape[0],np.amax(Y)+1),dtype=int) 
            for i in range (Y1.shape[0]): Y1[i:i+1][0,Y[i:i+1]]=1
            X_train = X[:int(X.shape[0]*train_size)]
            Y_train = Y1[:int(Y1.shape[0]*train_size)]
            X_test = X[int(X.shape[0]*train_size):]
            Y_test = Y1[int(Y1.shape[0]*train_size):]
            return ((X_train,Y_train),(X_test,Y_test))
      else :
        X= np.copy(dataset[0])
        Y= np.copy(dataset[1])
        X_train = X[:int(X.shape[0]*train_size)]
        Y_train = Y[:int(Y.shape[0]*train_size)]
        X_test = X[int(X.shape[0]*train_size):]
        Y_test = Y[int(Y.shape[0]*train_size):]
        return ((X_train,Y_train),(X_test,Y_test))

    
    #add a layer to the network
    def add(self, layer):
        self.layers.append(layer)

    #define the loss function
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    #make a prediction on an element of the dataset 
    def predict(self, input_data):
        lenght = len(input_data)
        prediction = []
        for i in range(lenght):
            res = input_data[i]
            for l in self.layers:
                res = l.forward_propagation(res)
            prediction.append(res)
        return prediction

    #Validate image test
    def validate(self,file_path):
      image_array = cv2.imread(file_path, cv2.IMREAD_COLOR )
      img_size = 0
      for layer in self.layers :
        if isinstance(layer, ConvLayer) :
          img_size = layer.input_shape[0]
          break
      image_array = ResizeImage(image_array, img_size, img_size)
      image_array = ConvertToGray(image_array)
      image_array1 = NormalizeImage(image_array)
      image_array = np.expand_dims(image_array, axis = 0)
      image_array = np.expand_dims(image_array, axis = 3)
      tmp = self.predict(image_array)
      result = np.argmax(tmp)
      return result

    #Returns the value of the updated learning rate according to the learning progress
    # mode == 1 User chose the Learning_Rate
    #mode == 2 The learning rate is automatically chosen
    #indice_epoch == the indice of the current epoch
    #epoch == number total of epoch
    # epoch + indice_epoch only make sense if the mode == 2
    def Learning_rate_schedule(self,mode,epoch,indice_epoch,Learning_rate):
     if(mode == 1):
        return Learning_rate
     elif(mode == 2):
        if indice_epoch > epoch/2 :
          lr = round(Learning_rate,2)
          return lr
        elif indice_epoch > epoch/3:
          lr = round(Learning_rate*0.8,2)
          return lr
        elif indice_epoch > epoch/4:
          lr = round(Learning_rate*0.6,2)
          return lr
        else :
          lr = round(Learning_rate*0.2,2)
          return lr

    #check if the prediction made on a given is correct or not
    def verification_of_prediction(self,x,y,i):
      predict=np.argmax(self.predict(x[i:i+1]))
      result=np.argmax(y[i:i+1])
      if result == predict : return 1
      else : return 0

    #save information from the neural network
    def information_network(self) :
      Network_info = list()
      layer = list()
      information = list()
      parametre = list()
      for l in self.layers :
         if isinstance (l,FullyConnectedLayer) :
          layer.append("Fully")
          parametre.append(l.input_size)
          parametre.append(l.output_size)
          information.append(l.weight)
          information.append(l.bias)
         if isinstance (l,FlattenLayer) :
          layer.append("Flatten")
         if isinstance (l,ActivationLayer) :
          layer.append("Activation")
          if (l.activ == sigmoid) :
            parametre.append("sigmoid")
            parametre.append("sigmoid_prime")
          elif (l.activ == rectified_linear_unit) :
            parametre.append("rectified_linear_unit")
            parametre.append("rectified_linear_unit_prime")
          elif (l.activ == tanh) : 
            parametre.append("tanh")
            parametre.append("tanh_prime")
          elif (l.activ == softmax) :
             parametre.append("softmax")
             parametre.append("softmax_prime")
         if  isinstance (l,ConvLayer) :
          layer.append("Conv")
          parametre.append(l.input_shape)
          parametre.append(l.kernel_shape)
          parametre.append(l.layer_depth)
          information.append(l.weights)
          information.append(l.bias)
      Network_info.append(layer)
      Network_info.append(parametre)
      Network_info.append(information)
      return Network_info


    #Evaluate the performance of a neural network from a test set
    def evaluate(self,x,y):
      Y_test =[]
      Y_pred =[]
      result=0
      for i in range(x.shape[0]):
        if self.verification_of_prediction(x,y,i)==1 :
          result = result + 1
        Y_pred.append(np.argmax(self.predict(x[i:i+1])))
        Y_test.append(np.argmax(y[i:i+1]))
      test_accuarcy= result / float(x.shape[0])
      taille=y.shape[1]
      matrice_conf=np.zeros((taille,taille),dtype=int)
      for test,pred in zip(Y_test,Y_pred) : matrice_conf[test:test+1][0,pred]+=1
      return (test_accuarcy, matrice_conf)

    #train the neural network
    def fit(self, x_train, y_train,epochs,mode,learning_rate):
      lenght = len(x_train)
      for i in range(epochs):
          err = 0
          l_rate=self.Learning_rate_schedule(mode,epochs,epochs-i+1,learning_rate)
          for j in range(lenght):
              output = x_train[j]
              for layer in self.layers:
                  output = layer.forward_propagation(output)
              err += self.loss(y_train[j], output)
              training_accuracy = training_accuracy + self.verification_of_prediction(x_train,y_train,j)
              error = self.loss_prime(y_train[j], output)
              for layer in reversed(self.layers):
                  error = layer.backward_propagation(error, l_rate)
          training_accuracy = training_accuracy / float(lenght)
          err = err / lenght
          print('epoch %d/%d   error=%f   training_accuracy=%f' % (i+1, epochs, err, training_accuracy))
