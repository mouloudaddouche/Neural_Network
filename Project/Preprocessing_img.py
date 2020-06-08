import numpy as np
import os
import pickle
import cv2
import random
from sklearn.model_selection import train_test_split
from skimage.transform import rotate, warp, AffineTransform
import globals
def get_number_instances(file_path) :
  """
    This function gives the number of instances in a dataset.

    Arguments :
    file_path -- the path of the csv file

    Return :
    nb_instance -- the number of instances
  """
  f = open(file_path,'r')
  f.readline()
  categories = list()
  nb_instance =0
  for ligne in f.readlines() :
    nb_instance = nb_instance+1
  f.close()
  return nb_instance

def get_number_class(file_path):
  """
    This function gives the number of classes in a dataset.

    Arguments :
    file_path -- the path of the csv file

    Return :
    len(categories) -- the number of classes
  """
  f = open(file_path,'r')
  f.readline()
  categories = list()
  for ligne in f.readlines() :
    var=ligne.strip().split(',')
    classe=var[1] # class
    #*****retrieve all the classes in the categories list***************
    if(classe not in categories):
      categories.append(classe)
  f.close()
  return len(categories)

def RotateLeft(image):
  """
    This fucntion does a left rotation of an image.

    Arguments :
    image -- the matrix representing an image

    Return :
    res -- the image after the left rotation
  """
  rows,cols = image.shape[:2]  #retrieve the dimensions of the image
  M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
  #(col/2,rows/2) ==> image rotation center,
  #M ==> the coordinates of the center
  res = cv2.warpAffine(image,M,(cols,rows))
  return res

def RotateRight(image):
  """
    This fucntion does a right rotation of an image.

    Arguments :
    image -- the matrix representing an image

    Return :
    res -- the image after the right rotation
  """
  rows,cols = image.shape[:2]  #retrieve the dimensions of the image
  M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
  #(col/2,rows/2) ==> image rotation center,
  #M ==> the coordinates of the center
  res = cv2.warpAffine(image,M,(cols,rows))
  return res

def Mirror(image):
  """
    This fucntion does a horizontal flip.

    Arguments :
    image -- the matrix representing an image

    Return :
    np.fliplr(image) -- the image afeter the flip
  """
  return np.fliplr(image)

def FlipVertical(image):
  """
    This function does a vertical flip.

    Arguments :
    image -- the matrix representing an image

    Return :
    res -- the image after the flip
  """

  rows,cols = image.shape[:2]  #retrieve the dimensions of the image
  M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
  #(col/2,rows/2) ==> image rotation center,
  #M ==> the coordinates of the center
  res = cv2.warpAffine(image,M,(cols,rows))
  return res

def RandomTransformation (image):
  """
  Random transformation of an image, for data augmentation
  Take an input image, then apply one of the transformations to it
  """
  r = random.randint(1,4);
  options = {
            1: RotateLeft,
            2: RotateRight,
            3: FlipVertical,
            4: Mirror,
          }
  return options[r](image)

def ImagetoArray(path, img):
  """
  Represent the image as a matrix
  """
  image_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR )
  return image_array

def ResizeImage(img_array, length, height):
  """
  Resize an Image
  """
  new_array = cv2.resize(img_array, (length, height))
  return new_array

def ConvertToGray(img_array):
  """
  Convert an image to shades of gray
  """
  imageBW = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY )
  return imageBW

def NormalizeImage(image):
  """
  Normalize an image
  Take an input matrix, and return a matrix containing values between 0 and 1
  """
  norm_img = np.zeros((int (globals.length_m), int (globals.height_m)))                  # initialize a matrix 30*30 a 0
  norm_img = cv2.normalize(image, norm_img, 0, 30, cv2.NORM_MINMAX, dtype = cv2.CV_32F)
  image_normalised = norm_img/255
  #Divide all elements of the matrix by 255 to have values between 0 and 1
  return image_normalised


def DataAugmentation(training_data1, csv_path1, path1, categories1):
  """
  we apply a random transformation to the images of the dataset,
  then we normalize them, and we add them in the training_data
  """
  f = open(csv_path1,'r')
  f.readline()

  for ligne in f.readlines() :
    #recover the image path and its classes
    var=ligne.strip().split(',')
    img_p=var[0]   # image's name
    classe=var[1]  # class
    id_classe = categories1.index(classe)
    #**************#Represent the image as a matrix****************
    image_array = ImagetoArray(path1, img_p)
    ##*************Random image transformation*********************
    img = RandomTransformation(image_array)
    ##*************Resize image************************************
    img_res = ResizeImage(img, int (globals.length_m), int (globals.height_m))
    ##**************Conversion to shades of gray*******************
    img_nb = ConvertToGray(img_res)
    ##**************Normalization**********************************
    image_normalised = NormalizeImage(img_nb)
    #***************filling the matrix data/class******************
    # add the image to the data matrix
    training_data1.append([image_normalised, id_classe])

  f.close()
  return training_data1

def Preprocessing (csv_path2, path2):
  """
  For each image in the dataset:
    we get its path and class
    we represent it as a matrix
    we resize it
    we convert it to shades of gray
    we normalize it
    then we add it to the training_data
  """
  #open the csv file (2 columns: the 1st for the image path and the second for the class)
  f = open(csv_path2,'r')
  #categories is a list which will contain all the classes of the dataset
  categories = list()
  training_data = []   #contains [matrix_image, class]

  f.readline()

  for ligne in f.readlines() :
    #recover the image path and its classes
    var=ligne.strip().split(',')
    img=var[0]    #path image
    classe=var[1] #class

    #**************Retrieve all the classes in the category list****************
	#we represent each class by an integer, to do this, we store the classes in the "categories" list,
	#then we represent each class by its index in the "categories" list
    if(classe not in categories):
      categories.append(classe)
    id_classe = categories.index(classe)
    #**************Represent the image as a matrix******************************
    image_array = ImagetoArray(path2, img)
    ##*************Resize image*************************************************
    new_array = ResizeImage(image_array, int(globals.length_m), int(globals.height_m))
    ##**************Conversion to shades of gray********************************
    imageNB = ConvertToGray(new_array)
    ##**************Normalization***********************************************
    image_normalised = NormalizeImage(imageNB)

    #***************filling the matrix data/class*******************************
    #add the image to the data matrix
    training_data.append([image_normalised, id_classe])

  f.close()
  #***************Data augmentation*********************************************
  training_data = DataAugmentation(training_data, csv_path2, path2, categories)
  return training_data
