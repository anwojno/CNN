import os
import csv
import shutil
import configparser
from sklearn.model_selection import train_test_split
from random import shuffle

""" data preprocessing """

first_class = 'cardiomegaly'
second_class = 'no cardiomegaly'
 
config = configparser.ConfigParser()
config.read('config.ini')

<<<<<<< HEAD

dataset_path = config['Paths']['dataset']
#test_file_path = 'D:/Users/Anna/CNN/CNN_rtg_cardiomegaly/test_list.txt'
#training_file_path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/train_val_list.txt'
new_dataset_path = 'Card dataset'

training_set_path = new_dataset_path + '/training_set'
ilness_path_training = new_dataset_path + '/training_set/' + first_class
no_ilness_path_training = new_dataset_path + '/training_set/' + second_class
test_set__path = new_dataset_path + '/test_set'
ilness_path_test = new_dataset_path + '/test_set/' + first_class
no_ilness_path_test = new_dataset_path + '/test_set/' + second_class

=======

dataset_path = config['Paths']['dataset']
#test_file_path = 'D:/Users/Anna/CNN/CNN_rtg_cardiomegaly/test_list.txt'
#training_file_path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/train_val_list.txt'
new_dataset_path = 'D:/datasets/Card dataset'

training_set_path = new_dataset_path + '/training_set'
ilness_path_training = new_dataset_path + '/training_set/' + first_class
no_ilness_path_training = new_dataset_path + '/training_set/' + second_class
test_set__path = new_dataset_path + '/test_set'
ilness_path_test = new_dataset_path + '/test_set/' + first_class
no_ilness_path_test = new_dataset_path + '/test_set/' + second_class

"""
training_set_path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/training_set'
ilness_path_training = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/training_set/' + first_class
no_ilness_path_training = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/training_set/' + second_class
test_set__path = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/test_set'
ilness_path_test = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/test_set/' + first_class
no_ilness_path_test = 'C:/Users/Anna/CNN/CNN_rtg_cardiomegaly/dataset/test_set/' + second_class
"""

>>>>>>> d9c674051069125b32634da1ae7491fd5d755c2e

os.mkdir(training_set_path)
os.mkdir(test_set__path)
os.mkdir(ilness_path_training)
os.mkdir(no_ilness_path_training)
os.mkdir(ilness_path_test)
os.mkdir(no_ilness_path_test)


images_filename_list = []
cardiomegaly_list = []
no_cardiomegaly_list = []

for filename in os.listdir(dataset_path):
    images_filename_list.append(filename)

print(images_filename_list)

my_dict = {}
with open('Data_Entry_2017.csv', mode='r') as file:
    reader = csv.reader(file, delimiter=',')
    my_dict = {rows[0]:rows[1] for rows in reader}

#print(my_dict)              
"""
number_training_ilness = 1
number_training_no_illness = 1
number_test_ilness = 1
number_test_no_illness = 1
"""
cardiomegaly_count = 0
"""
for k, v in my_dict.items():
    #print(k, v)
    if k in images_filename_list:
        if 'Cardiomegaly' in v:
            #if v == 'Cardiomegaly':
                cardiomegaly_list.append(k)
                cardiomegaly_count += 1
        elif 'No Finding' in v:
            if k.rstrip()[-5] != '0':
                #print(k.rstrip())
                continue
            else:
                no_cardiomegaly_list.append(k)
        else:
            no_cardiomegaly_list.append(k)

        
"""        
for k, v in my_dict.items():
    #print(k, v)
    if k in images_filename_list:
        if 'Cardiomegaly' in v:
            if v == 'Cardiomegaly':
                cardiomegaly_list.append(k)
                cardiomegaly_count += 1
        elif 'No Finding' in v:
            if k.rstrip()[-5] != '0':
                #print(k.rstrip())
                continue
            else:
                no_cardiomegaly_list.append(k)
        else:
            #no_cardiomegaly_list.append(k)
            continue

        
shuffle(no_cardiomegaly_list)

testset_list = cardiomegaly_list + no_cardiomegaly_list[0: len(cardiomegaly_list)]

shuffle(testset_list)
 

print(cardiomegaly_count)
print(len(my_dict))
print(len(cardiomegaly_list))
print(len(no_cardiomegaly_list))

X_train, X_test = train_test_split(testset_list, test_size=0.2)

print(len(X_train))
print(len(X_test))


for filename in X_train:
    if 'Cardiomegaly' in my_dict[filename]:
        shutil.copyfile(dataset_path + '/' + filename, ilness_path_training + '/' + filename)
        #number_training_ilness += 1
        print('cardio', my_dict[filename])
    else:
        shutil.copyfile(dataset_path + '/' + filename, no_ilness_path_training + '/' + filename)
        #number_training_no_illness += 1
        print('no_cardio', my_dict[filename])
        
for filename in X_test:
    if 'Cardiomegaly' in my_dict[filename]:
        shutil.copyfile(dataset_path + '/' + filename, ilness_path_test + '/' + filename)
        #number_test_ilness += 1
        print('cardio', my_dict[filename])
    else:
        shutil.copyfile(dataset_path + '/' + filename, no_ilness_path_test + '/' + filename)
        #number_test_no_illness += 1
        print('no_cardio', my_dict[filename])
