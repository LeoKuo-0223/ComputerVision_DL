# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:50:21 2021

@author: leo90
"""

import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile


def main():
    base_dir = r'C:\Users\leo90\ComputerVision_DL\HW2\kagglecatsanddogs_3367a'
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)
    #make folders for the images of cats and dogs and assign to train, test validation
    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)
    
    #start to splut dataset
    split_size_train = 0.8
    split_size_test = 0.1
    split_size_val = 0.1
    #directory
    CAT_SOURCE_DIR = "kagglecatsanddogs_3367a/PetImages/Cat/"
    TRAINING_CATS_DIR = "kagglecatsanddogs_3367a/train/cats/"
    TESTING_CATS_DIR = "kagglecatsanddogs_3367a/test/cats/"
    VALIDATION_CATS_DIR = "kagglecatsanddogs_3367a/validation/cats/"
    DOG_SOURCE_DIR = "kagglecatsanddogs_3367a/PetImages/Dog/"
    TRAINING_DOGS_DIR = "kagglecatsanddogs_3367a/train/dogs/"
    TESTING_DOGS_DIR = "kagglecatsanddogs_3367a/test/dogs/"
    VALIDATION_DOGS_DIR = "kagglecatsanddogs_3367a/validation/dogs/"
    
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, VALIDATION_CATS_DIR, 
               split_size_train, split_size_test, split_size_val)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, VALIDATION_DOGS_DIR, 
               split_size_train, split_size_test, split_size_val)
    

def split_data(SOURCE, TRAINING, TESTING, VALIDATION,SPLIT_SIZE_Train,SPLIT_SIZE_Test,SPLIT_SIZE_val):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename    #file dir
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE_Train)
    testing_length = int(len(files) *SPLIT_SIZE_Test)
    validation_length = int(len(files)*SPLIT_SIZE_val)
    
    shuffled_set = random.sample(files, len(files))
    
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[training_length:(training_length+testing_length)]
    validation_set = shuffled_set[-validation_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)
        
    for filename in validation_set:
        this_file = SOURCE + filename
        destination = VALIDATION+ filename
        copyfile(this_file, destination)
        
        
    print(len(os.listdir('kagglecatsanddogs_3367a/train/cats/')))
    print(len(os.listdir('kagglecatsanddogs_3367a/test/cats/')))
    print(len(os.listdir('kagglecatsanddogs_3367a/validation/cats/')))
    print(len(os.listdir('kagglecatsanddogs_3367a/train/dogs/')))
    print(len(os.listdir('kagglecatsanddogs_3367a/test/dogs/')))
    print(len(os.listdir('kagglecatsanddogs_3367a/validation/dogs/')))
        
if __name__=='__main__':
    main()