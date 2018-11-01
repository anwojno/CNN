# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:12:55 2018

@author: Anna
"""
import os
import csv
import shutil

""" data preprocessing """

first_class = 'cardiomegaly'
second_class = 'no cardiomegaly'
 
dataset_path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset'
test_file_path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/test_list.txt'
training_file_path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/train_val_list.txt'

training_set_path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/training_set'
ilness_path_training = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/training_set/' + first_class
no_ilness_path_training = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/training_set/' + second_class
test_set__path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/test_set'
ilness_path_test = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/test_set/' + first_class
no_ilness_path_test = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/test_set/' + second_class

"""
os.mkdir(training_set_path)
os.mkdir(test_set__path)
os.mkdir(ilness_path_training)
os.mkdir(no_ilness_path_training)
os.mkdir(ilness_path_test)
os.mkdir(no_ilness_path_test)
"""

my_dict = {}
with open('C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/Data_Entry_2017.csv', mode='r') as file:
    reader = csv.reader(file)
    my_dict = {rows[0]:rows[1] for rows in reader}
              

number_training_ilness = 1
number_training_no_illness = 1
number_test_ilness = 1
number_test_no_illness = 1


for filename in os.listdir(dataset_path):
    test_list = open(test_file_path, 'r')
    for line in test_list: 
        if filename == line.rstrip():
            if 'No Finding' in my_dict[filename]:
                pass
            elif 'Cardiomegaly' in my_dict[filename]: 
                shutil.move(dataset_path + '/' + filename, ilness_path_test + '/' + first_class + '.' + str(number_test_ilness) + '.png')
                number_test_ilness += 1
            else:
                shutil.move(dataset_path + '/' + filename, no_ilness_path_test + '/' + second_class + '.' + str(number_test_no_illness) + '.png')
                number_test_no_illness += 1
            break

             
for filename in os.listdir(dataset_path):
    training_list = open(training_file_path, 'r')
    for line in training_list: 
        if filename == line.rstrip():
            if 'No Finding' in my_dict[filename]:
                pass
            elif 'Cardiomegaly' in my_dict[filename]: 
                shutil.move(dataset_path + '/' + filename, ilness_path_training + '/' + first_class + '.' + str(number_training_ilness) + '.png')
                number_training_ilness += 1
            else:
                shutil.move(dataset_path + '/' + filename, no_ilness_path_training + '/' + second_class + '.' + str(number_training_no_illness) + '.png')
                number_training_no_illness += 1
            break