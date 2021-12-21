# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 02:05:35 2021

@author: leo90
"""

import random
import tensorflow as tf

from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, utils
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import ResNet50_GUI as ui
import warnings
warnings.filterwarnings('ignore')


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.Show_structure)
        self.pushButton_2.clicked.connect(self.Show_tensorboard)
        self.pushButton_3.clicked.connect(self.Show_test)
        self.pushButton_4.clicked.connect(self.Compare)
        
        self.DATASET_PATH  = 'sample' # 資料路徑
        self.IMAGE_SIZE = (224, 224) # 影像大小
        self.NUM_CLASSES = 2 # 影像類別數
        self.BATCH_SIZE = 80
        self.NUM_EPOCHS = 5 #EPOCH
        # 凍結網路層數
        self.FREEZE_LAYERS = 2
        
        # 模型輸出儲存的檔案
        self.WEIGHTS_FINAL = 'model-resnet50-final.h5'
        
        # 透過 data augmentation 產生訓練與驗證用的影像資料
        self.train_datagen_augment = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,
                                            shear_range=0.2,zoom_range=0.2,channel_shift_range=10,
                                            horizontal_flip=True,fill_mode='nearest')
        self.TRAINING_DIR = "kagglecatsanddogs_3367a/train/"
        self.VALIDATION_DIR = "kagglecatsanddogs_3367a/validation/"
        self.TEST_DIR = "kagglecatsanddogs_3367a/test/"
        self.train_datagen = ImageDataGenerator(rescale=1.0/255.)
        self.train_batches = self.train_datagen.flow_from_directory(self.TRAINING_DIR,target_size=self.IMAGE_SIZE,
                                                          interpolation='bicubic',class_mode='categorical',classes=['dogs', 'cats'],
                                                          shuffle=True,batch_size=self.BATCH_SIZE)
        
        self.valid_datagen = ImageDataGenerator(rescale=1.0/255.)
        self.valid_batches = self.valid_datagen.flow_from_directory(self.VALIDATION_DIR,target_size=self.IMAGE_SIZE,
                                                          interpolation='bicubic',class_mode='categorical',classes=['dogs', 'cats'],
                                                          shuffle=False,batch_size=self.BATCH_SIZE)
        self.test_datagen = ImageDataGenerator(rescale=1.0/255.)
        self.test_batches = self.valid_datagen.flow_from_directory( self.TEST_DIR,target_size=self.IMAGE_SIZE,
                                                          interpolation='bicubic',class_mode='categorical',classes=['dogs', 'cats'],
                                                          shuffle=False,batch_size=self.BATCH_SIZE)
        # self.test_datagen_origin = ImageDataGenerator()
        # self.test_batches_origin = self.valid_datagen.flow_from_directory( self.TEST_DIR,target_size=self.IMAGE_SIZE,
        #                                                   interpolation='bicubic',class_mode='categorical',classes=['dogs', 'cats'],
        #                                                   shuffle=False,batch_size=self.BATCH_SIZE)
        
        # 以訓練好的 ResNet50 為基礎來建立模型，
        # 捨棄 ResNet50 頂層的 fully connected layers
        self.Resnet50_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                       input_shape=(self.IMAGE_SIZE[0],self.IMAGE_SIZE[1],3))
        self.x = self.Resnet50_model.output
        self.x = layers.Flatten()(self.x)
        
        # 增加 DropOut layer
        self.x = layers.Dropout(0.5)(self.x)
        
        # 增加 Dense layer，以 softmax 產生個類別的機率值
        self.output_layer = layers.Dense(self.NUM_CLASSES, activation='softmax', name='softmax')(self.x)
        
        self.Resnet50_final_Model = Model(inputs=self.Resnet50_model.input, outputs=self.output_layer)
        for layer in self.Resnet50_final_Model.layers[:self.FREEZE_LAYERS]:
            layer.trainable = False
        for layer in self.Resnet50_final_Model.layers[self.FREEZE_LAYERS:]:
            layer.trainable = True
        # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
        self.Resnet50_final_Model.compile(optimizers.Adam(learning_rate=1e-5),
                          loss='categorical_crossentropy', metrics=['acc'])
        
        
    def Show_structure(self):
        print(self.Resnet50_final_Model.summary())
       
    def Train_model(self):
        self.Resnet50_final_Model.fit(self.train_batches,steps_per_epoch = self.train_batches.samples // self.BATCH_SIZE,
                                validation_data = self.valid_batches,
                                validation_steps = self.valid_batches.samples // self.BATCH_SIZE,
                                epochs = self.NUM_EPOCHS)
        self.Resnet50_final_Model.save(self.WEIGHTS_FINAL)
    
    def Show_tensorboard(self):
        plt.figure().set_size_inches(15, 15)
        epoch_accuracy = img.imread('epoch_accuracy.jpg')
        plt.subplot(2, 1, 1)
        plt.title('Accuracy', fontsize=9)
        plt.axis('off')
        plt.imshow(epoch_accuracy)

        epoch_loss = img.imread('epoch_loss.jpg')
        plt.subplot(2, 1, 2)
        plt.title('Loss', fontsize=9)
        plt.axis('off')
        plt.imshow(epoch_loss)
        plt.show()
        
    def Show_test(self):
        textboxValue = self.lineEdit.text()
        textboxValue = int(textboxValue)
        original_Prediction = np.loadtxt("Prediction.txt").reshape(2500, 2)
        if original_Prediction.size==0:
            model = load_model('model-resnet50-final.h5')
            print("Predicting the images in Testing folder")
            original_Prediction = model.predict(self.test_batches)
            print("Finish !!")
            a_file = open("Prediction.txt", "w")
            for row in original_Prediction:
                np.savetxt(a_file, row)
            a_file.close()
        else:
            # print(self.Prediction.shape)
            # print(self.Prediction)
            batch_index = int(textboxValue/80)
            image_index = textboxValue%80
            plt.imshow(self.test_batches[batch_index][0][image_index])
            plt.axis('off')
            if original_Prediction[textboxValue][0]>original_Prediction[textboxValue][1]:
                plt.title('dogs', fontsize=12)
            else:
                plt.title('cats', fontsize=12)
            plt.show()
    
    def Compare(self):
        Compare_result = img.imread('Comparision.jpg')
        plt.axis('off')
        plt.imshow(Compare_result)
        plt.show()
        
        
        
    
    def Eval(self):
        model = load_model('model-resnet50-final.h5')
        scores = model.evaluate(self.test_batches)
        print('\n準確率=', scores[1])  #準確率= 0.9955999851226807
        model_aug = load_model('model-resnet50-final_aug.h5')
        scores_aug = model_aug.evaluate(self.test_batches)
        print('\nAugmentation準確率=', scores_aug[1])#Augmentation準確率= 0.9911999702453613
        plt.title('Augmentation Comparision')
        labels = ['Before','After']
        prediction = [scores[1],scores_aug[1]]
        plt.ylim(0.9,1,0.05)
        plt.bar(labels,prediction , width=0.5,align='center')
        plt.show()
    
# main
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())