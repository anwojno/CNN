# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:12:55 2018

@author: Anna
"""

# backend - TensorFlow

from keras.models import Sequential  # initializing nn as a sequence of layers
from keras.layers import Convolution2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense  # fully connected layer
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator  # for preprocessing training and test images   
from skimage.io import imread  # for loading image for prediction
import numpy as np  # for preprocessing image used for prediction


dataset_path = "D:/datasets/Card dataset/"


""" initializing CNN """
classifier = Sequential()


""" first convolutional layer """
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 1))  
""" Args:
    number of filters: can be 32, 64, 128 ... 
    number of rows and number of columns of filter
    input_shape: to convert images to the same fixed size, (256,256, 3) - for full sized coloured images
"""

classifier.add(BatchNormalization())  # resistance to vanishing gradient during training

classifier.add(Activation("relu"))


""" first pooling layer """
classifier.add(MaxPooling2D(pool_size=(2,2)))


""" second convolutional layer """
classifier.add(Convolution2D(64, (3, 3))  
# don't need to include input_shape since we work on pooled feature maps

classifier.add(BatchNormalization())

classifier.add(Activation("relu"))


""" second pooling layer """
classifier.add(MaxPooling2D(pool_size=(2,2)))


""" flattening layer """
# spatial structer is thanks to convolution 
# where we put in feature maps the highest number of specific feature
classifier.add(Flatten())


""" fully connected hidden layer """
classifier.add(Dense(units = 128, activation = 'relu'))
""" Args:
     units: number of nodes in the hidden layer, best to choose number between number of input nodes and the number of outut nodes
     activation: activation function, relu to return the probability of each class
"""
classifier.add(Dropout(0.5))  
#  at each training stage, individual nodes are dropped out of the net 
# neurons develop co-dependency amongst each other during training 
# which curbs the individual power of each neuron leading to over-fitting of training data


""" fully connected output layer """
classifier.add(Dense(units = 1, activation = 'sigmoid'))

""" Args:
     units: number of nodes in the output layer - the predicted probability of one class
     activation: activation function, sigmoid because we have binary outcome, for more than 2 outcomes - softmax
"""


""" compiling the CNN """
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
""" Args:
     optimizer: choosing stochastic gradient descent algorithm
     loss: choosing loss function, crossentropy - used for classification problems, for more than 2 outcomes - caterogical_crossentropy
     metric: choosing performance metric 
"""


""" fitting the CNN to the images """

# image augmentation - generates batches of image data with real-time data augmentation, 
# shifting existing images, rotating them etc. to reduce overfitting
# code from Keras documentation - Example of using - flow from directory

# augmentation - training set
train_datagen = ImageDataGenerator(rescale=1./255)  # all pixel values will be between 0 and 1
#        shear_range=0.2, # to apply random transvections
#        zoom_range=0.2,
#        horizontal_flip=True


# rescaling images of our test set
test_datagen = ImageDataGenerator(rescale=1./255)


# creating training set from augmentation applied on our training set
training_set = train_datagen.flow_from_directory(
               dataset_path + 'training_set',
               target_size=(64, 64),
               color_mode="grayscale",  # or "rgb"
               batch_size=16,
               class_mode='binary') # for more than 2 - categorical 

                                                 
# creating test set from augmentation applied on our test set
test_set = test_datagen.flow_from_directory(
            dataset_path + 'test_set',
            target_size=(64, 64),
            color_mode="grayscale",   # or "rgb"
            batch_size=16,
            class_mode='binary')


# fitting CNN model on our training set and testing it's performance on our test set
# conda install pillow in your env or pip install pillow if you're not using env
classifier.fit_generator(training_set,
                         steps_per_epoch=(1607//16), #  number of samples of your dataset divided by the batch size
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=(638//16))  # number of samples of your validation dataset divided by the batch size

                         

""" making new prediction """

class_labels = {v: k for k, v in training_set.class_indices.items()} # creates a dictionary with mapping, ex. 0 = cat, 1 = dog
 
test_image = imread(dataset_path + 'single_prediction/cardiomegaly5.png')
test_image = np.resize(test_image, (64,64,1))
test_image = np.expand_dims(test_image, axis=0) 
""" Adding another dimension to image, because predict function uses 4 dimensions - last one for batch number 
    Args:
     image: the image we're predicting
     axis: the position of the index of the dimension we're adding
"""
if (np.max(test_image) > 1):
    test_image = test_image/255.0  # preprocessing the image so that the pixels have value between 0 and 1
 
prediction = classifier.predict_classes(test_image) # result is an array with one value, ex 1 for dog, 0 for cat
 
print(class_labels[prediction[0][0]])


###############################################################################################################################
#################################################################################################################################

# for more than 2 classess

# number of output units in the final layer = number of classes

# output_activation = 'softmax'

# loss = 'categorical_crossentropy'

# y_matrix must be onehotencoded labels.  

# class_mode = 'categorical'

