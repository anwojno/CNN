# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:12:55 2018

@author: Anna
"""

# backend - TensorFlow

from keras.models import Sequential  # initializing nn as a sequence of layers
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense  # fully connected layer
from keras.preprocessing.image import ImageDataGenerator  # for preprocessing images  

""" initializing CNN """
classifier = Sequential()


""" convolution layer """
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))  
""" Args:
    number of filters: can be 32, 64, 128 ... 
    number of rows and number of columns of filter
    input_shape: to convert images to the same fixed size, (256,256, 3) - for full sized coloured images
    activation function: rectyfier - to have non-liniearity
"""


""" pooling layer """
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
train_datagen = ImageDataGenerator(
        rescale=1./255,  # all pixel values will be between 0 and 1
        shear_range=0.2, # to apply random transvections
        zoom_range=0.2,
        horizontal_flip=True)

# rescaling images of our test set
test_datagen = ImageDataGenerator(rescale=1./255)


# creating training set from augmentation applied on our training set
training_set = train_datagen.flow_from_directory(
              'dataset/training_set',
               target_size=(64, 64),
               batch_size=32,
               class_mode='binary') # for more than 2 - categorical ?

# creating test set from augmentation applied on our test set
test_set = test_datagen.flow_from_directory(
           'dataset/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

# fitting CNN model on our training set and testing it's performance on our test set
# conda install pillow in your virtual env or pip install pillow if you're not using virtual env
classifier.fit_generator(training_set,
                         steps_per_epoch=8000, # number of images in epoch
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)  # number of images in test set