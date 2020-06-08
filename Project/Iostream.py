#!/usr/bin/python
# coding : utf8
import h5py
import csv
import os
import json
import pickle
import shutil
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.io import arff
import os, shutil, sys
from fullyConnected_layer import FullyConnectedLayer
from convolutional_layer import ConvLayer
from flatten_layer import FlattenLayer
from activation_layer import ActivationLayer
from losses import *
from network import Network


from activations import tanh, tanh_prime, sigmoid, sigmoid_prime, rectified_linear_unit, rectified_linear_unit_prime, softmax
from preprocessing_text import get_attributes, is_nominal


matrix_saves_path = "./Matrix_saves/"
model_saves_path = "./Model_saves/"

# global variables for save
dataset_name = ""
input_array = []
output_array = []

### TEMPRARY FILES MANAGEMENT ###

#CREATION D'UN DOSSIER TEMPORAIRE (POUR LES OUTILS DE VISUALISATION)
def create_tmp_folder () :
    if not os.path.exists('tmp') :
        os.mkdir('tmp')

#SUPPRESSION DU CONTENU DU DOSSIER TEMPORAIRE
def remove_tmp_folder_content () :
    if os.path.exists('tmp') :
        shutil.rmtree('tmp')

#SUPPRESSION DU DOSSIER TEMPORAIRE
def remove_tmp_folder () :
    if os.path.exists('tmp') :
        remove_tmp_folder_content()
        os.rmdir('tmp')

#SUPPRESSION DES FICHIERS GENERES AVEC MATPLOTLIB
def remove_generated_file (file_path) :
    if os.path.exists(file_path) :
        os.remove(file_path)

### DATASET LOADER ###

#CHARGER LES DATASETS ARFF
def dataset_arff_loader (file_path) :
    name_file = file_path.split('/')[-1]
    ext = name_file.split('.')[-1]
    raw_data = arff.loadarff(file_path)
    df = pd.DataFrame(raw_data[0])
    attributes = get_attributes(df)
    #pour l'encodage
    for att in attributes :
        if df[att].dtypes == "object" :
            df[att] = df[att].str.decode("utf-8")
    #REMPLACER LES ? PAR NP.NAN
    #CREATION D'UNE LISTE ATTRIBUTES QUI CONTIENT QUE LES ATTRIBUTS NOMINAUX
    attributes = []
    atts = get_attributes(df)
    for att in atts:
        if is_nominal(att, df) :
            attributes.append(att)
    for att in attributes:
        df[att]= df[att].replace('?', np.nan)
    df = df.replace('?', np.nan)
    return df


def save_dataset(training_data, dico, dataset_name):
  #sauvegarde le dataset pretraite  + un fichier des classes 
  pickle_out = open(dataset_name+".pickle","wb")
  pickle.dump(training_data, pickle_out)
  pickle_out.close()

  pickle_out = open(dataset_name+"dico_class.pickle","wb")
  pickle.dump(dico, pickle_out)
  pickle_out.close()


# CHARGEMENT DATASET PRETRAITE
def load_dataset(path_dataset, path_class):
  pickle_in_datset = open(path_dataset,"rb")
  pickle_in_dico = open(path_class,"rb")
  dic = pickle.load(pickle_in_dico) 
  if (type(dic) != type(dict())):
    return 0
  return pickle.load(pickle_in_datset), dic 

#CHARGER LES DATASETS CSV
def dataset_csv_loader (file_path) :
    name_file = file_path.split('/')[-1]
    ext = name_file.split('.')[-1]
    df = pd.read_csv(file_path, sep=',')
    return df

def loadColumnDataset(path_Xpickel):
  #Charger un fichier .pickel (pour charger un dataset deja pretraite auparavant)
  pickle_in = open(path_Xpickel,"rb")
  return pickle.load(pickle_in)

### PREPROCESSED DATASET SAVE ###

def saveColumn1Dataset(file_path):
    if '.csv' in file_path:
        #Retourne un dictionnaire ou la cle : la classe, la valeur: le numero qui lui correspond
        f = open(file_path,'r') 
        f.readline()
        dico = {}
        i=0
        for ligne in f.readlines() :
          var=ligne.strip().split(',')
          classe=var[1] 
          if(classe not in dico):
            dico[classe]=i
            i+=1
        f.close()
    else:
        name_file = file_path.split('/')[-1]
        ext = name_file.split('.')[-1]
        raw_data = arff.loadarff(file_path)
        df = pd.DataFrame(raw_data[0])
        attributes = list(df.columns.values)
        att_class=attributes[df.shape[1]-1]
        dico={}
        etique_class=0
        for val in df[att_class]:
          if not pd.isnull(val):
            if val not in dico: 
                dico[val]=etique_class
                etique_class=etique_class+1
    return dico

def saveColumn2Dataset(training_data, dico, dataset_name):
    #sauvegarde le dataset pretraite  + un fichier des classes 
    pickle_out = open(dataset_name+".pickle","wb")
    pickle.dump(training_data, pickle_out)
    pickle_out.close()
    pickle_out = open(dataset_name+"dico_class.pickle","wb")
    pickle.dump(dico, pickle_out)
    pickle_out.close()


### NEURAL NETWORK SAVE ###
def manual_save_model_neural_network (neural_network, neural_network_name) :
    """
        This function save a neural network model in a Json file.

        Arguments :
        neural_network -- a Network object.
        neural_network_name -- the neural network name for he file's name.
    """

    # Get the number of layers
    network_info = neural_network.information_network()
    length = len (network_info[0])
    # dictionary for json save
    data = {}
    index_parameter = 0
    for i in range (length) :
        if network_info[0][i] == "Fully" :

            data['Fully_' + str(i)] = []
            data ['Fully_' + str(i)].append ({
                'input_size' : network_info[1][index_parameter],
                'output_size' : network_info[1][index_parameter + 1],
            })
            index_parameter += 2

        if network_info[0][i] == "Flatten" :

            data['Flatten_' + str(i)] = []

        if network_info[0][i] == "Activation" :

            data['Activation_' + str(i)] = []
            data ['Activation_' + str(i)].append ({
                'activation' : network_info [1][index_parameter],
                'activation_prime' : network_info[1][index_parameter + 1]
            })
            index_parameter += 2

        if network_info[0][i] == "Conv" :

            data['Conv_' + str(i)] = []
            data['Conv_' + str(i)].append ({
                'input_shape' : network_info[1][index_parameter],
                'kernel_shape' : network_info[1][index_parameter + 1],
                'layer_depth' : network_info[1][index_parameter + 2],
            })
            index_parameter += 3

    # Checking if directory exists
    if not os.path.isdir(model_saves_path) :
        os.mkdir ('Model_saves')

    path = model_saves_path + neural_network_name + ".json"
    # Checking if file exists
   # path = os.path.expanduser (path)
   # if not os.path.exists (path) :
   #     pass
   # else :
   #     root, ext = os.path.splitext (os.path.expanduser (path))
   #     directory = os.path.dirname (root)
   #     file_name = os.path.basename (root)
   #     candidate = file_name + ext
   #     index = 0
   #     ls = set (os.listdir(directory))
   #     while candidate in ls :
   #         candidate = "{}_{}{}".format (file_name, index, ext)
   #         index += 1
   #     path = os.path.join (directory, candidate)



    with open(path, 'w') as outfile:
        json.dump(data, outfile)

def save_matrix_neural_network (neural_network, neural_network_name) :
    """
        This function creates a hdf5 file that save the biais and the weights of the current neural network.

        Arguments :
        neural_network -- a list representing the neural network, it has both bias and weight matrix
        path -- file's path

        Return :
        hdf5 file.
    """

    # Checking if directory exists
    if not os.path.isdir(matrix_saves_path) :
        os.mkdir ('Matrix_saves')

    path = matrix_saves_path + neural_network_name + ".h5"
    # Checking if file exists
    #path = os.path.expanduser (path)
    #if not os.path.exists (path) :
    #    pass
    #else :
    #    root, ext = os.path.splitext (os.path.expanduser (path))
    #    directory = os.path.dirname (root)
    #    file_name = os.path.basename (root)
    #    candidate = file_name + ext
    #    index = 0
    #    ls = set (os.listdir(directory))
    #    while candidate in ls :
    #        candidate = "{}_{}{}".format (file_name, index, ext)
    #        index += 1
    #    path = os.path.join (directory, candidate)


    with h5py.File (path , "w") as save :
        group_fully = save.create_group ('Fully')
        group_conv = save.create_group ('Conv')
        i=0
        for l in neural_network.layers :
         if isinstance (l,FullyConnectedLayer) :
                group_fully.create_dataset ('FullyW_' + str(i), data = l.weight)
                group_fully.create_dataset ('FullyB_' + str(i), data = l.bias)
    

         if  isinstance (l,ConvLayer) :
                group_conv.create_dataset ('ConvW_' + str(i), data = l.weights)
                group_conv.create_dataset ('ConvB_' + str(i), data = l.bias)
        
         i+=1


def automatic_save_neural_network (neural_network, time) :
    pass

### NEURAL NETWORK LOADER ###
def neural_network_loader (hdf5_file_path, json_file_path) :
    """
        This function loads a json file and a hd5 file. Then it creates the model and the matrix of the neural network.

        Arguments :
        hdf5_file_path -- The hdf5 file's path
        json_file_path -- The json file's path

        Return :
        an object of type network
    """
    network_info = list()
    layer = list()
    information = list()
    parameter = list()
    # Loading json file and  hdf5 file
    with open (json_file_path) as json_file, h5py.File (hdf5_file_path, 'r') as hdf5File :
        data = json.load (json_file)
        group1 = hdf5File.get ('Fully')
        group2 = hdf5File.get ('Conv')
        len_groupFully=len(group1)
        len_groupConv=len(group2)
        groupFully=0
        groupConv=0
        for i in range (len(data)):
          for key in data :
           if (key[-1] == str(i)) :
            if key[:-2] == "Fully" :
                groupFully+=2
                for x in data[key] :
                    layer.append ("Fully")
                    parameter.append (x['input_size'])
                    parameter.append (x['output_size'])
                    information.append (np.array (group1.get("FullyW_" + key[-1:])))
                    information.append (np.array (group1.get("FullyB_" + key[-1:])))
            if key[:-2] == "Flatten" :
                    layer.append ("Flatten")
            if key[:-2] == "Activation" :
                for x in data[key] :
                    layer.append ("Activation")
                    parameter.append (x['activation'])
                    parameter.append (x['activation_prime'])
            if key[:-2] == "Conv" :
                groupConv+=2
                for x in data[key] :
                    layer.append("Conv")
                    parameter.append (x['input_shape'])
                    parameter.append (x['kernel_shape'])
                    parameter.append (x['layer_depth'])
                    information.append (np.array (group2.get("ConvW_" + key[-1:])))
                    information.append (np.array (group2.get("ConvB_" + key[-1:])))
    # Creation of a network object
    if (groupFully != len_groupFully) or (groupConv != len_groupConv) :
      return 0
    else :
        net=Network()
        i=0
        j=0
        k=0
        for l in layer :
          if l == "Conv" :
            net.add(ConvLayer(parameter[i],parameter[i+1],parameter[i+2]))
            net.layers[k].weights = np.copy(information[j])
            net.layers[k].bias = np.copy(information[j+1])
            i+=3
            j+=2
            k+=1
          if l == "Flatten" :
            net.add(FlattenLayer())
            k+=1
          if l == "Activation" :
              if (parameter[i] == "sigmoid") :
                net.add(ActivationLayer(sigmoid,sigmoid_prime))
              if (parameter[i] == "tanh") :
                net.add(ActivationLayer(tanh,tanh))
              if (parameter[i] == "rectified_linear_unit") :
                net.add(ActivationLayer(rectified_linear_unit,rectified_linear_unit_prime))
              if (parameter[i] == "softmax") :
                net.add(ActivationLayer(softmax,softmax_prime))
              k+=1
              i+=2
          if l == "Fully" :
              net.add(FullyConnectedLayer(parameter[i],parameter[i+1]))
              net.layers[k].weight = np.copy(information[j])
              net.layers[k].bias = np.copy(information[j+1])
              k+=1
              j+=2
              i+=2
        net.use(mean_squared_error, mean_squared_error_prime)
    return net


### LOADING FILE TO TEST ###
def image_loader (file_path) :
    """
        This function loads an Image and stores it in a matrix where each element is an Image's pixel.

        Arguments :
        file_path -- the Image's path

        Return :
        image_array -- 2D matrix where each element is the BGR of a pixel
    """
    image_array = cv2.imread(file_path, cv2.IMREAD_COLOR )
    return image_array

if __name__ == "__main__" :

    #### IMAGE LOADER
    #print "TEST IMAGE_LOADER"
    #image_matrix = Image_loader ("/home/vataye/Images/Wallpapers/japan.jpg")
    #print image_matrix

    #### CSV LOADER
    #print "TEST CSV_LOADER"
    #dataset_csv_loader ("/home/vataye/Documents/addresses.csv")
    #print myDataSet

    #### SAVE HDF5 FILE
    pass
    ## SAVE JSON FILE
    #manual_save_model_neural_network (matrix_test, "mymodel")

    ## LOADING DATA
    #neural_network_loader ("./Matrix_saves/test1_1", "./mymodel")
