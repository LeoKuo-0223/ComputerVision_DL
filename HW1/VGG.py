# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:01:19 2021

@author: leo90
"""
import random
import tensorflow as tf

from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, utils
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import GUI_VGG as ui
import warnings
warnings.filterwarnings('ignore')


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):

        super().__init__()
        (self.Feature_train, self.Label_train), (self.Feature_test,
                                                 self.Label_test) = datasets.cifar10.load_data()
        self.Label_train = self.Label_train.flatten()
        self.Label_test = self.Label_test.flatten()
        self.Feature_train_fl = self.Feature_train.astype('float32')
        self.Feature_test_fl = self.Feature_test.astype('float32')
        self.Label_train_oneHot = tf.keras.utils.to_categorical(
            self.Label_train, 10)
        self.Label_test_oneHot = tf.keras.utils.to_categorical(
            self.Label_test, 10)
        # print(self.Label_test[0])

        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.Show_traindataset)
        self.pushButton_2.clicked.connect(self.Show_Hyperparameter)
        self.pushButton_3.clicked.connect(self.Show_Model_structure)
        self.pushButton_4.clicked.connect(self.TrainModel)
        self.pushButton_5.clicked.connect(self.Loadmodel)
        self.vgg_layers_16 = [
            tf.keras.Input(shape=(32, 32, 3)),
            # stack1
            layers.Conv2D(64, kernel_size=[3, 3],
                          padding='same', activation=tf.nn.relu),
            layers.Conv2D(64, kernel_size=[3, 3],
                          padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # stack2
            layers.Conv2D(128, kernel_size=[
                          3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(128, kernel_size=[
                          3, 3], padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # stack3
            layers.Conv2D(256, kernel_size=[
                          3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[
                          3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[
                          1, 1], padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # stack4
            layers.Conv2D(512, kernel_size=[
                          3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[
                          3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[
                          1, 1], padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # stack5
            layers.Conv2D(512, kernel_size=[
                          3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[
                          3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[
                          1, 1], padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            layers.Flatten(),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(10, activation='softmax'),

        ]

        # build model vgg16
        self.vgg16_model = Sequential(self.vgg_layers_16)
        # self.vgg16_model.build(input_shape=[None,32,32,3])
        self.Learning_Rate = 0.0001
        self.batch_size = 200

    def Show_traindataset(self):
        randomList = random.sample(range(0, 49999), 9)
        count = 1
        for i in randomList:
            plt.subplot(3, 3, count)
            plt.title(self.indexToLabels(self.Label_train[i]), fontsize=9)
            plt.axis('off')
            plt.imshow(self.Feature_train[i])
            count = count+1
        plt.show()
        # print(self.Feature_train.shape)
        # print(self.Label_train.shape[0])

    def indexToLabels(self, index):
        return {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }.get(index, 'error')

    def Show_Model_structure(self):
        self.vgg16_model.summary()

    def Show_Hyperparameter(self):
        print("Hyperparameter: ")
        print("batch size: ", self.batch_size)
        print("Learing Rate: ", self.Learning_Rate)
        print("Optimizer: Adam")

    def TrainModel(self):
        # self.vgg16_model.compile(optimizers.Adam(lr=self.Learning_Rate), loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        # self.vgg16_model.fit(x=self.Feature_train_fl, y=self.Label_train_oneHot, validation_split=0.2,
        #                 epochs=20, verbose=2,batch_size=self.batch_size)
        # self.vgg16_model.save('VGG16_Cifar10_model_1.h5')     #將模型儲存至 HDF5檔案中
        # print("VGG16_Cifar10_model_1.h5 模型儲存完畢!")
        # self.Eval()

        # already train the model on google colab and load model in the following function
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

    def Eval(self):
        scores = self.vgg16_model.evaluate(
            self.Feature_test_fl, self.Label_test_oneHot)
        print('\n準確率=', scores[1])

    def Loadmodel(self):
        textboxValue = self.lineEdit.text()
        textboxValue = int(textboxValue)
        model = load_model('VGG16_Cifar10_model_1.h5')
        # randomNumlist = random.sample(range(0, 9999), 1)
        # randomNum = randomNumlist[0]
        # test_input = tf.reshape(self.Feature_test_fl[randomNum],[-1,32,32,3])
        test_input = tf.reshape(
            self.Feature_test_fl[textboxValue], [-1, 32, 32, 3])

        prediction = model.predict(test_input)
        prediction = prediction.flatten()
        # print(prediction)
        outcome = 0
        for i in range(len(prediction)):
            if((prediction[i]) > outcome):
                outcome = prediction[i]
                predict_Num = i
        # predicted picture
        plt.figure().set_size_inches(15, 5)
        plt.subplot(1, 2, 1)
        # plt.figure(figsize=(20, 20))
        plt.title(self.indexToLabels(predict_Num), fontsize=12)
        plt.axis('off')
        # plt.imshow(self.Feature_test[randomNum])
        plt.imshow(self.Feature_test[textboxValue])

        # bar chart
        labels = ["airplane", "automobile", "bird", "cat",
                  "deer", "dog", "frog", "horse", "ship", "truck", ]
        plt.subplot(1, 2, 2)
        plt.bar(labels, prediction, width=0.5, align='center')
        plt.xticks(rotation=45)
        plt.show()
        


# main
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
